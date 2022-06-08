function [x] = dc_motor(u)
s = serialport('COM3', 115200)
write(s, [254 u], 'uint8')
write(s, 255, 'uint8')
data = read(s, 2, 'uint8')
F = data(:,1)*256;
r = data(:,2);
x = (F+r)*8.982025
flush (s)
clear s
end