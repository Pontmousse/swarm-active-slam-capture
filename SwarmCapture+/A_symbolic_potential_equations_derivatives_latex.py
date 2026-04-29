from sympy import *  # import everything

# Define symbols
x, y, z = symbols('x y z')
xk, yk, zk = symbols('x_k y_k z_k')
Rflk = symbols('R_flk')
nuflk = symbols('nu_flk')

# Define the potential using sympy.sqrt instead of np.sqrt
Pot_flk = nuflk / 2 * (sqrt((x - xk)**2 + (y - yk)**2 + (z - zk)**2) - Rflk)**2

diffx = Pot_flk.diff(x)
diffy = Pot_flk.diff(y)
diffz = Pot_flk.diff(z)

print('\n')
print(latex(Pot_flk))
print('\n')
print(latex(diffx))
print('\n')
print(latex(diffy))
print('\n')
print(latex(diffz))
print('\n')




print('\n')

# Define symbols
x, y, z = symbols('x y z')
xj, yj, zj = symbols('x_j y_j z_j')
Rant = symbols('R_ant')
nuant = symbols('nu_ant')

# Define the potential using sympy.sqrt instead of np.sqrt
Pot_ant = nuant / 2 * (1/sqrt((x - xk)**2 + (y - yk)**2 + (z - zk)**2) - 1/Rant)**2

diffx = Pot_ant.diff(x)
diffy = Pot_ant.diff(y)
diffz = Pot_ant.diff(z)

print('\n')
print(latex(Pot_ant))
print('\n')
print(latex(diffx))
print('\n')
print(latex(diffy))
print('\n')
print(latex(diffz))
print('\n')
