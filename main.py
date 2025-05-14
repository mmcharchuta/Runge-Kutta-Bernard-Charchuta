import matplotlib.pyplot as plt
import os

p1 = 8.8
p2 = 440
p3 = 100
d1 = 1.375*(10**-14)
d2 = 1.375*(10**-4)
d3 = 3*(10**-5)
k1 = 1.925*(10**-4)
k2 = 10**5
k3 = 1.5*(10**5)
value_siRNA = 0.02
value_PTEN_off = 0
value_no_DNA_damage = 0.1

def f_p53(p53, mdmn): 
    return p1 - d1*p53*(mdmn**2)

def f_mdmcyto(p53, mdmcyto, pten, siRNA=False, no_DNA_damage=False):
    siRNA_factor = value_siRNA if siRNA else 1
    DNA_damage_factor = value_no_DNA_damage if no_DNA_damage else 1
    return p2*siRNA_factor*(p53**4)/((p53**4) + (k2**4)) - k1*(k3**2)/((k3**2) + (pten**2))*mdmcyto - d2*DNA_damage_factor*mdmcyto

def f_mdmn(mdmn, mdmcyto, pten, no_DNA_damage=False):
    if no_DNA_damage:
        return k1*(k3**2)/((k3**2) + (pten**2))*mdmcyto - d2*value_no_DNA_damage*mdmn
    else:
        return k1*(k3**2)/((k3**2) + (pten**2))*mdmcyto - d2*mdmn

def f_pten(pten, p53, pten_off=False):
    if not pten_off:
        return p3*(p53**4)/((p53**4) + (k2**4)) - d3*pten
    else:
        return p3*value_PTEN_off*(p53**4)/((p53**4) + (k2**4)) - d3*pten


def runge_kutta(p53, mdcyto, mdmn, pten, h, siRNA=False, pten_off=False, no_DNA_damage=False):
    k1_p53 = f_p53(p53, mdmn)
    k1_mdmcyto = f_mdmcyto(p53, mdcyto, pten, siRNA)
    k1_mdmn = f_mdmn(mdmn, mdcyto, pten)
    k1_pten = f_pten(pten, p53, pten_off)

    k2_p53 = f_p53(p53 + h/2*k1_p53, mdmn + h/2*k1_mdmn)
    k2_mdmcyto = f_mdmcyto(p53 + h/2*k1_p53, mdcyto + h/2*k1_mdmcyto, pten + h/2*k1_pten, siRNA)
    k2_mdmn = f_mdmn(mdmn + h/2*k1_mdmn, mdcyto + h/2*k1_mdmcyto, pten + h/2*k1_pten)
    k2_pten = f_pten(pten + h/2*k1_pten, p53 + h/2*k1_p53, pten_off)

    k3_p53 = f_p53(p53 + h/2*k2_p53, mdmn + h/2*k2_mdmn)
    k3_mdmcyto = f_mdmcyto(p53 + h/2*k2_p53, mdcyto + h/2*k2_mdmcyto, pten + h/2*k2_pten, siRNA)
    k3_mdmn = f_mdmn(mdmn + h/2*k2_mdmn, mdcyto + h/2*k2_mdmcyto, pten + h/2*k2_pten)
    k3_pten = f_pten(pten + h/2*k2_pten, p53 + h/2*k2_p53, pten_off)

    k4_p53 = f_p53(p53 + h*k3_p53, mdmn + h*k3_mdmn)
    k4_mdmcyto = f_mdmcyto(p53 + h*k3_p53, mdcyto + h*k3_mdmcyto, pten + h*k3_pten, siRNA)
    k4_mdmn = f_mdmn(mdmn + h*k3_mdmn, mdcyto + h*k3_mdmcyto, pten + h*k3_pten)
    k4_pten = f_pten(pten + h*k3_pten, p53 + h*k3_p53,  pten_off)

    p53 += (k1_p53 + 2*k2_p53 + 2*k3_p53 + k4_p53) * h / 6
    mdcyto += (k1_mdmcyto + 2*k2_mdmcyto + 2*k3_mdmcyto + k4_mdmcyto) * h / 6
    mdmn += (k1_mdmn + 2*k2_mdmn + 2*k3_mdmn + k4_mdmn) * h / 6
    pten += (k1_pten + 2*k2_pten + 2*k3_pten + k4_pten) * h / 6
    return p53, mdcyto, mdmn, pten


    

def main():
    # DANE:
    h = 6 # time step in minutes
    iterations = int(48*60/h) # number of iterations during 48 hours
    p53_0 = 10
    mdmcyto_0 = 2000
    mdmn_0 = 10000
    pten_0 = 2000
    print(p53_0, mdmcyto_0, mdmn_0, pten_0)

    output_folder = "plots"
    os.makedirs(output_folder, exist_ok=True)

    # Conditions:
    conditions = { 
        "Basic" : (False, False, True), # siRNA, PTEN_off, no_DNA_damage
        "Damaged DNA" : (False, False, False),
        "Tumor" : (False, True, False),
        "Therapy" : (True, False, False),
    }

    for condition, (siRNA, pten_off, no_DNA_damage) in conditions.items():
        p53 = p53_0
        mdmcyto = mdmcyto_0
        mdmn = mdmn_0
        pten = pten_0
        time_values = []
        p53_values = []
        mdmcyto_values = []
        mdmn_values = []
        pten_values = []
        print("Condition:", condition)
        for i in range(iterations):
            time = i * h
            time_values.append(time)
            p53_values.append(p53)
            mdmcyto_values.append(mdmcyto)
            mdmn_values.append(mdmn)
            pten_values.append(pten)

            p53, mdmcyto, mdmn, pten = runge_kutta(p53, mdmcyto, mdmn, pten, h, siRNA, pten_off, no_DNA_damage)
        plt.plot(time_values, p53_values, label="p53")
        plt.plot(time_values, mdmcyto_values, label="MDMcyto")
        plt.plot(time_values, mdmn_values, label="MDMn")
        plt.plot(time_values, pten_values, label="PTEN")
        plt.xlabel("Time [min]")
        plt.ylabel("Concentration [nM]")
        plt.title(f"Concentration of proteins in 48 hours ({condition})")
        plt.legend()
        plt.grid(True)
        filename = os.path.join(output_folder, f"{condition.replace(' ', '_')}.png")
        plt.savefig(filename)
        plt.show()
        plt.close()
        break 



if __name__ == "__main__":
    main()