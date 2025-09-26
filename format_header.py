import sys
data = open(sys.argv[1], 'rb').read()
hex_values = [hex(b) for b in data]
lines = []
current_line = 'const unsigned char ' + sys.argv[3] + '[] = {'
for i, hex_val in enumerate(hex_values):
    if len(current_line + hex_val + ',') > 120 and i > 0:
        lines.append(current_line.rstrip(',') + ',')
        current_line = '  '
    current_line += hex_val + ','
lines.append(current_line.rstrip(',') + '};')
open(sys.argv[2], 'w').write('\n'.join(lines))
