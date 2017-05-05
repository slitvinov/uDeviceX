d = '~/workspace/source_codes/RBC/rbc_uq/tools/tran_son_tay/data';
f = 'post.new';
p = dlmread(sprintf('%s/%s.txt', d, f), '', 1, 0);
sh = p(:, 5); p = p(sh == 5, :);
r =  p(:, 1);
go = p(:, 2);
gi = p(:, 3);
gc = p(:, 4);
sh = p(:, 5);
fr = p(:, 6); fru  = p(:, 7);
a  = p(:, 8);  au  = p(:, 9);
b  = p(:, 10); bu  = p(:, 11);
c  = p(:, 12); cu  = p(:, 13);
th = p(:, 14); thu = p(:, 15);
el = p(:, 16); elu = p(:, 17);
a_ = p(:, 18); au_ = p(:, 19);
b_ = p(:, 20); bu_ = p(:, 21);
c_ = p(:, 22); cu_ = p(:, 23);

set(0,'DefaultTextFontSize', 30)
set(0, 'DefaultAxesFontSize', 30)
set(0, 'DefaultLineLineWidth', 5)
set(0, 'DefaultLineColor', 'black')

figure(1)
x = go; y = gi; z = gc;
xt = unique(x); yt = unique(y); zt = unique(z);
dp = 0.5; s = 10000;
hold on
q1 = fr; q2 = 0.218203; v = (q1-q2)/q2; scatter3(x-2*dp, y, z, s, v, 'Marker', '.')
q1 = a_; q2 = 1.900000; v = (q1-q2)/q2; scatter3(x-1*dp, y, z, s, v, 'Marker', '.')
q1 = c_; q2 = 0.525000; v = (q1-q2)/q2; scatter3(x+0*dp, y, z, s, v, 'Marker', '.')
q1 = th; q2 = 12.80000; v = (q1-q2)/q2; scatter3(x+1*dp, y, z, s, v, 'Marker', '.')
q1 = el; q2 = 125.0000; v = (q1   )/q2; scatter3(x+2*dp, y, z, s, v, 'Marker', '.')
for i=1:length(xt)
    for k=1:length(zt)
        line([xt(i) xt(i)],   [yt(1) yt(end)], [zt(k) zt(k)])
    end
end
hold off

grid on
xlabel('go'); ylabel('gi'); zlabel('gc')
xlim([min(x)-3*dp max(x)+3*dp])
ylim([min(y)      max(y)     ])
zlim([min(z)      max(z)     ])
xticks(xt); yticks(yt); zticks(zt)
colormap('jet'); caxis([-0.2 0.2]); colorbar
% campos([-21.4473 -191.4344 66.9401])
campos([-20 -200 50])

%%
my_save_fig(sprintf('%s/%s.fig', d, f), 40, 40);
my_save_fig(sprintf('%s/%s.pdf', d, f), 40, 40);
