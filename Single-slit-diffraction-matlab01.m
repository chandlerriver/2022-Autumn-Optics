lamda=500e-9; %波长
N=1; %缝数 ，可以随意更改变换
a=2e-4;D=5;d=5*a;
ym=2*lamda*D/a;xs=ym;%屏幕上y的范围
n=1001;%屏幕上的点数
ys=linspace(-ym,ym,n);%定义区域
for i=1:n
  sinphi=ys(i)/D;
  alpha=pi*a*sinphi/lamda;
  beta=pi*d*sinphi/lamda;
  B(i,:)=(sin(alpha)./alpha).^2.*(sin(N*beta)./sin(beta)).^2;
  B1=B/max(B);
end
NC=256; %确定灰度的等级
Br=(B/max(B))*NC;
subplot(1,2,1)
image(xs,ys,Br);
colormap(hot(NC)); %色调处理
subplot(1,2,2)
plot(B1,ys,'k');
