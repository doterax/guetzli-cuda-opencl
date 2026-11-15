/*
 * OpenCL/CUDA edition implementation of ButteraugliComparator.
 *
 * Author: strongtu@tencent.com
 *         ianhuang@tencent.com
 *         chriskzhou@tencent.com
 */
#include <algorithm>
#include <stdint.h>
#include <vector>

#if defined(__USE_OPENCL__) || defined(__USE_CUDA__)

#include "clguetzli.h"
#include "opsin_block.h"
#ifdef __USE_CUDA__
#include "cuguetzli.h"
#include "ocu.h"
#endif

namespace guetzli
{
    ButteraugliComparatorEx::ButteraugliComparatorEx(const int width, const int height,
        const std::vector<uint8_t>* rgb,
        const float target_distance, ProcessStats* stats)
        : ButteraugliComparator(width, height, rgb, target_distance, stats)
    {
        if (MODE_CPU != g_mathMode)
        {
            rgb_orig_opsin.resize(3);
            rgb_orig_opsin[0].resize(width * height);
            rgb_orig_opsin[1].resize(width * height);
            rgb_orig_opsin[2].resize(width * height);

            const float* lut = Srgb8ToLinearTableF();
            
            for (int c = 0; c < 3; ++c) {
                for (int y = 0, ix = 0; y < height_; ++y) {
                    for (int x = 0; x < width_; ++x, ++ix) {
                        rgb_orig_opsin[c][ix] = lut[rgb_orig_[3 * ix + c]];
                    }
                }
            }
            ::butteraugli::OpsinDynamicsImage(width_, height_, rgb_orig_opsin);
        }
    }

    ButteraugliComparatorEx::~ButteraugliComparatorEx()
    {
#ifdef __USE_OPENCL__
        clCompareDestroy(cl_ctx_);
#endif
#ifdef __USE_CUDA__
        cuCompareDestroy(cu_ctx_);
#endif
    }

    void ButteraugliComparatorEx::Compare(const OutputImage& img)
    {
		if (MODE_CPU_OPT == g_mathMode)
		{
			std::vector<std::vector<float> > rgb0 = rgb_orig_opsin;

			std::vector<std::vector<float> > rgb(3, std::vector<float>(width_ * height_));
			img.ToLinearRGB(&rgb);
			::butteraugli::OpsinDynamicsImage(width_, height_, rgb);
			std::vector<float>().swap(distmap_);
			comparator_.DiffmapOpsinDynamicsImage(rgb0, rgb, distmap_);
			distance_ = ::butteraugli::ButteraugliScoreFromDiffmap(distmap_);
		}
#ifdef __USE_OPENCL__
        else if (MODE_OPENCL == g_mathMode)
        {
            // The reference image stays on the device for the comparator's lifetime;
            // the output image is converted to linear RGB on the device.
            if (!cl_ctx_)
            {
                cl_ctx_ = clCompareCreate(rgb_orig_opsin[0].data(), rgb_orig_opsin[1].data(), rgb_orig_opsin[2].data(), width_, height_);
            }
            distmap_.resize((size_t)width_ * height_);
            clCompare(cl_ctx_, img, comparator_.step(), distmap_.data());
            distance_ = ::butteraugli::ButteraugliScoreFromDiffmap(distmap_);
        }
#endif
#ifdef __USE_CUDA__
        else if (MODE_CUDA == g_mathMode)
        {
            if (!cu_ctx_)
            {
                cu_ctx_ = cuCompareCreate(rgb_orig_opsin[0].data(), rgb_orig_opsin[1].data(), rgb_orig_opsin[2].data(), width_, height_);
            }
            distmap_.resize((size_t)width_ * height_);
            cuCompare(cu_ctx_, img, comparator_.step(), distmap_.data());
            distance_ = ::butteraugli::ButteraugliScoreFromDiffmap(distmap_);
        }
#endif
		else
		{
			ButteraugliComparator::Compare(img);
		}
    }

    void ButteraugliComparatorEx::StartBlockComparisons()
    {
        if (MODE_CPU == g_mathMode)
        {
            ButteraugliComparator::StartBlockComparisons();
            return;
        }

        // Mask(orig, orig) depends only on the original image: compute it once and
        // reuse it for every block-comparison round (SelectFrequencyMasking runs 2-3x).
        if (mask_xyz_orig_.empty())
        {
            std::vector<std::vector<float> > dummy(3);
            ::butteraugli::Mask(rgb_orig_opsin, rgb_orig_opsin, width_, height_, &mask_xyz_orig_, &dummy);
        }
        mask_xyz_ = mask_xyz_orig_; // FinishBlockComparisons() clears mask_xyz_

        // The per-block lists only feed the GPU block-zeroing kernel; also built once.
        if (MODE_CPU_OPT == g_mathMode || block_lists_ready_)
        {
            return;
        }

        const int width = width_;
        const int height = height_;
        const int factor_x = 1;
        const int factor_y = 1;

        const int block_width = (width + 8 * factor_x - 1) / (8 * factor_x);
        const int block_height = (height + 8 * factor_y - 1) / (8 * factor_y);
        const int num_blocks = block_width * block_height;

        const float* lut = Srgb8ToLinearTableF();

        imgOpsinDynamicsBlockList.resize(num_blocks * 3 * kDCTBlockSize);
        imgMaskXyzScaleBlockList.resize(num_blocks * 3);
        for (int block_y = 0, block_ix = 0; block_y < block_height; ++block_y)
        {
            for (int block_x = 0; block_x < block_width; ++block_x, ++block_ix)
            {
                float* curR = &imgOpsinDynamicsBlockList[block_ix * 3 * kDCTBlockSize];
                float* curG = curR + kDCTBlockSize;
                float* curB = curG + kDCTBlockSize;

                for (int iy = 0, i = 0; iy < 8; ++iy) {
                    for (int ix = 0; ix < 8; ++ix, ++i) {
                        int x = std::min(8 * block_x + ix, width - 1);
                        int y = std::min(8 * block_y + iy, height - 1);
                        int px = y * width + x;

                        curR[i] = lut[rgb_orig_[3 * px]];
                        curG[i] = lut[rgb_orig_[3 * px + 1]];
                        curB[i] = lut[rgb_orig_[3 * px + 2]];
                    }
                }

                OpsinDynamicsBlock8x8((float(*)[kOpsinBlockSize])curR);

                int xmin = block_x * 8;
                int ymin = block_y * 8;

                imgMaskXyzScaleBlockList[block_ix * 3] = mask_xyz_[0][ymin * width_ + xmin];
                imgMaskXyzScaleBlockList[block_ix * 3 + 1] = mask_xyz_[1][ymin * width_ + xmin];
                imgMaskXyzScaleBlockList[block_ix * 3 + 2] = mask_xyz_[2][ymin * width_ + xmin];
            }
        }
        block_lists_ready_ = true;
    }

    void ButteraugliComparatorEx::FinishBlockComparisons() {
        // Clears mask_xyz_; the cached mask and block lists are kept for the next round.
        ButteraugliComparator::FinishBlockComparisons();
    }
    
    double ButteraugliComparatorEx::CompareBlock(const OutputImage& img, int off_x, int off_y, const coeff_t* candidate_block, const int comp_mask) const
    {
        double err = ButteraugliComparator::CompareBlock(img, off_x, off_y, candidate_block, comp_mask);
        return err;
    }
}

#endif