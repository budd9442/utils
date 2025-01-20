# import hashlib
# key = [hashlib.sha256(b'GOUGH').hexdigest()[i] for i in [4,5,3,6,2,7,1,8]]
# s="picoCTF{1n_7h3_|<3y_of_" + ''.join([((str(k))) for k in key] ) + "}"
# print(s)

from calendar import c
from factordb.factordb import FactorDB
def crackRSA(c,n,e):
    f= FactorDB(n)
    p,q = f.connect()
    ph = (p-1)*(q-1)
    d = pow(e,-1,ph)  # gmpy2.invert(e, ph)
    plaintext = pow(c,d,n)
    return bytearray.fromhex(format(plaintext, 'x')).decode()

c=240986837130071017759137533082982207147971245672412893755780400885108149004760496
n=831416828080417866340504968188990032810316193533653516022175784399720141076262857
e=65537

print(crackRSA(c,n,e))