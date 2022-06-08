function dydt = first_order(x)
u = x(1);
y = x(2);

tau =0.2439 ;
k = 17.515;

dydt = (-y + k*u)/tau;



end