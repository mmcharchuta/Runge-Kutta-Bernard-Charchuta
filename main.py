import matplotlib.pyplot as plt

p1 = 8.8
p2 = 440
p3 = 100
d1 = 1.375*10**-14
d2 = 1.375*10**-4
d3 = 3*10**-5
k1 = 1.925*10**-4
k2 = 10**5
k3 = 1.5*10**5


def f_p53(p53, mdmn):
    p53_dot = p1 - d1*p53*mdmn**2 
    return p53_dot

def f_mdmcyto(p53, mdmcyto, pten):
    mdmcyto_dot = p2*(p53**4)/(p53**4 + k2**4) - k1*(k3**2)/(k3**2 + pten**2) *mdmcyto - d2*mdmcyto
    return mdmcyto_dot

def f_mdmn(mdmn, mdmcyto, pten):
    mdmn_dot = k1*(k3**2)/(k3**2 + pten**2)*mdmcyto - d2*mdmn
    return mdmn_dot

def f_pten(pten, p53):
    pten_dot = p3*(p53**4)/(p53**4 + k2**4) - d3*pten
    return pten_dot


def runge_kutta(p53, mdcyto, mdmn, pten, h, siRNA=False, pten_off=False, no_DNA_damage=False):
    k1_p53 = f_p53(p53, mdmn)
    k1_mdmcyto = f_mdmcyto(p53, mdcyto, pten)
    k1_mdmn = f_mdmn(mdmn, mdcyto, pten)
    k1_pten = f_pten(pten, p53)

    k2_p53 = f_p53(p53 + h/2*k1_p53, mdmn + h/2*k1_mdmn)
    k2_mdmcyto = f_mdmcyto(p53 + h/2*k1_p53, mdcyto + h/2*k1_mdmcyto, pten + h/2*k1_pten)
    k2_mdmn = f_mdmn(mdmn + h/2*k1_mdmn, mdcyto + h/2*k1_mdmcyto, pten + h/2*k1_pten)
    k2_pten = f_pten(pten + h/2*k1_pten, p53 + h/2*k1_p53)

    k3_p53 = f_p53(p53 + h/2*k2_p53, mdmn + h/2*k2_mdmn)
    k3_mdmcyto = f_mdmcyto(p53 + h/2*k2_p53, mdcyto + h/2*k2_mdmcyto, pten + h/2*k2_pten)
    k3_mdmn = f_mdmn(mdmn + h/2*k2_mdmn, mdcyto + h/2*k2_mdmcyto, pten + h/2*k2_pten)
    k3_pten = f_pten(pten + h/2*k2_pten, p53 + h/2*k2_p53)

    k4_p53 = f_p53(p53 + h*k3_p53, mdmn + h*k3_mdmn)
    k4_mdmcyto = f_mdmcyto(p53 + h*k3_p53, mdcyto + h*k3_mdmcyto, pten + h*k3_pten)
    k4_mdmn = f_mdmn(mdmn + h*k3_mdmn, mdcyto + h*k3_mdmcyto, pten + h*k3_pten)
    k4_pten = f_pten(pten + h*k3_pten, p53 + h*k3_p53)

    p53 += (k1_p53 + 2*k2_p53 + 2*k3_p53 + k4_p53) * h / 6
    mdcyto += (k1_mdmcyto + 2*k2_mdmcyto + 2*k3_mdmcyto + k4_mdmcyto) * h / 6
    mdmn += (k1_mdmn + 2*k2_mdmn + 2*k3_mdmn + k4_mdmn) * h / 6
    pten += (k1_pten + 2*k2_pten + 2*k3_pten + k4_pten) * h / 6
    return p53, mdcyto, mdmn, pten


    

def main():
    # DANE:
    h = 1
    time = 48*60

    if siRNA:
        p2 *= 0.02
    if pten_off:
        p3 *= 0
    if no_DNA_damage:
        d2 *= 0.1
    



if __name__ == "__main__":
    main()