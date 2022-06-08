function in = localResetFcn_ref(in)

% randomize reference signal
%{
nodes = [1000, 1500,2000, 2500,3000];
k= randi([1,5],1);
blk = sprintf('ddpg_new_try1_sim/Desired \nWater Level');
h = 100*randn + 2000;
%}
%{
while h <= 0 || h >= 3500
    h = 200*randn + 1500;
end
%}
%in = setBlockParameter(in,blk,'Value',num2str(h));

% randomize initial height
%h = 3*randn + nodes(k);
%{
while h <= 0 || h >= 3500
    h = 3*randn + 1500;SS
end
%}
h= 3*randn + 1000;
blk = 'ddpg_new_try1_sim_quad/DC-Motor/H';
in = setBlockParameter(in,blk,'InitialCondition',num2str(h));
%{
blk='ddpg_new_try1_sim/DC-Motor/Uin';
h= 10*randn +70;
in= setBlockParameter(in, blk,'Value',num2str(h));
%}
end