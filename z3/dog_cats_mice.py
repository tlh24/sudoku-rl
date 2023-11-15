from z3 import *
ndogs = Int('ndogs')
ncats = Int('ncats')
nmice = Int('nmice')
s = Solver()
s.add(ndogs >= 1)
s.add(ncats >= 1)
s.add(nmice >= 1)
s.add(ndogs + ncats + nmice == 100)
s.add(10*ndogs + ncats + 0.25*nmice == 103)
print(s.check())
print(s.model())

