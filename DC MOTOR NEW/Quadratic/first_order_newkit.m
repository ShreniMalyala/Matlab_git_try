function dydt = first_order_newkit(x)
u = x(1);
y = x(2);

tau =0.5505;
k = 51.92;

dydt = (-y + k*u)/tau;



end