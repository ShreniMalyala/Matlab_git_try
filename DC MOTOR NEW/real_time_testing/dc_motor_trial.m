function [data] = dc_motor_trial(u)
persistent s
if isempty(s)
    s = serialport('COM3', 115200);
end
write(s, [254 u], 'uint8');
write(s, 255, 'uint8');
data = read(s, 2, 'uint8');
end

