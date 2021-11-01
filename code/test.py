import opengen as og
import casadi.casadi as cs

u = cs.SX.sym("u",5)
print(u)