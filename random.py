# mt_streamlit_full_interactive.py
import streamlit as st
import time
import numpy as np

# --------------------------
# MTMini – zjednodušený pseudonáhodný generátor
# --------------------------
class MTMini:
    def __init__(self, seed, n=4, w=8, m=2):
        self.n = n
        self.w = w
        self.m = m
        self.upper_mask = 0xF0
        self.lower_mask = 0x0F
        self.state = [(seed + i) % (1 << w) for i in range(n)]
        self.index = n

    def twist_step(self, i):
        s_i = self.state[i]
        s_next = self.state[(i + 1) % self.n]
        upper = s_i & self.upper_mask
        lower = s_next & self.lower_mask
        y = upper | lower
        yA = y >> 1
        xor_const = 0xB8 if y & 1 else 0
        yA_final = yA ^ xor_const
        new_val = self.state[(i + self.m) % self.n] ^ yA_final
        return {"i": i, "s_i": s_i, "s_next": s_next, "upper": upper,
                "lower": lower, "y": y, "yA": yA, "xor_const": xor_const,
                "new_val": new_val}

    def twist_detailed(self):
        new_state = []
        details = []
        for i in range(self.n):
            step = self.twist_step(i)
            new_state.append(step["new_val"])
            details.append(step)
        self.state = new_state
        self.index = 0
        return details

    def extract(self):
        if self.index >= self.n:
            self.twist_detailed()
        y = self.state[self.index]
        y ^= y >> 1
        y ^= (y << 1) & 0xB8
        self.index += 1
        return y

# --------------------------
# Pomocné funkcie
# --------------------------
def bits(x, w=8):
    return format(x, f'0{w}b')

def invert_temper(y):
    y_inv = y
    for _ in range(8):
        y_inv = y ^ ((y_inv << 1) & 0xB8)
    y_final = y_inv
    for _ in range(8):
        y_final = y_inv ^ (y_final >> 1)
    return y_final & 0xFF

# --------------------------
# Streamlit UI
# --------------------------
st.title("MTMini – interaktívne generovanie, predikcia a vizualizácia pseudonáhodnosti")

st.markdown("""
Tento program demonštruje pseudonáhodný generátor krok po kroku:

- MTMini generuje čísla, ktoré **vyzerajú náhodne**, ale sú deterministické.
- Ukážeme vnútorný stav, twist a temperovanie bitov.
- Vizualizujeme **rozptyl hodnôt** a priestor náhodnosti.
- Umožníme **predikciu ďalšieho čísla** po prelomení stavu a overenie jej správnosti.
""")

# Inicializácia
seed = st.number_input("Seed (celé číslo):", value=1, step=1)
if "mt" not in st.session_state:
    st.session_state.mt = MTMini(seed)
    st.session_state.outputs = []

mt = st.session_state.mt
outputs = st.session_state.outputs

st.subheader("Počiatočný stav generátora:")
for i, val in enumerate(mt.state):
    st.text(f"s[{i}] = {val} -> {bits(val)}")

# Slider rýchlosti krokovania
speed = st.slider("Rýchlosť animácie (sekundy na krok)", 0.1, 2.0, 0.5, 0.1)

# --------------------------
# Generovanie ďalšieho čísla
# --------------------------
if st.button("Generate next number"):
    next_val = mt.extract()
    outputs.append(next_val)
    st.text(f"Generated number: {next_val} -> {bits(next_val)}")
    
    # Zobrazenie detailného výpočtu twistu krok po kroku
    st.subheader("Detailný výpočet (Twist krok):")
    details = mt.twist_detailed()
    for step in details:
        st.text(f"Index {step['i']}: s[i]={step['s_i']} ({bits(step['s_i'])}), "
                f"s[i+1]={step['s_next']} ({bits(step['s_next'])}), "
                f"upper={step['upper']} ({bits(step['upper'])}), "
                f"lower={step['lower']} ({bits(step['lower'])}), "
                f"y={step['y']} ({bits(step['y'])}), y>>1={step['yA']} ({bits(step['yA'])}), "
                f"xor_const={step['xor_const']} ({bits(step['xor_const'])}), new_val={step['new_val']} ({bits(step['new_val'])})")
        time.sleep(speed)

# --------------------------
# Zobrazenie všetkých výstupov a vizualizácia
# --------------------------
if outputs:
    st.subheader("Všetky generované výstupy:")
    st.text(outputs)
    st.bar_chart(np.array(outputs))
    
    if len(outputs) >=3:
        st.subheader("Jednoduchá 3D textová vizualizácia priestoru hodnôt:")
        st.text(f"x={outputs[0]}, y={outputs[1]}, z={outputs[2]} (rozptyl náhodnosti)")

# --------------------------
# Predikcia / Prelomenie
# --------------------------
if st.button("Predict next number (prelomenie)"):
    if len(outputs) < mt.n:
        st.warning(f"Na predikciu je potrebných minimálne {mt.n} výstupov.")
    else:
        reconstructed_state = [invert_temper(val) for val in outputs[:mt.n]]
        st.subheader("Rekonštruovaný stav generátora z výstupov:")
        st.text(reconstructed_state)
        
        mt_reconstructed = MTMini(seed=0)
        mt_reconstructed.state = reconstructed_state
        mt_reconstructed.n = len(reconstructed_state)
        mt_reconstructed.index = len(outputs) % mt_reconstructed.n
        predicted_val = mt_reconstructed.extract()
        st.text(f"Predikované ďalšie číslo = {predicted_val} -> {bits(predicted_val)}")
        
        # Overenie
        next_val_actual = mt.extract()
        outputs.append(next_val_actual)
        st.subheader("Overenie predikcie s reálnym generátorom:")
        st.text(f"Skutočné ďalšie číslo = {next_val_actual} -> {bits(next_val_actual)}")
        st.success(f"Predikcia {'sedí' if predicted_val == next_val_actual else 'nesedí'}!")

# --------------------------
# Teoretické vysvetlenie rozptylu a náhody
# --------------------------
st.subheader("Teória rozptylu a pseudonáhodnosti")
st.markdown("""
- **Pseudonáhodnosť:** čísla vyzerajú náhodne, ale sú deterministické z počiatočného seedu.
- **Rozptyl hodnôt:** vizualizuje, ako sa výstupy generátora rozkladajú v priestore hodnôt.
- **Prelomenie:** ak poznáme vnútorný stav (alebo dostatok výstupov), vieme predpovedať ďalšie číslo.
- Každý krok twistu kombinuje horné a dolné bity, posúva ich a robí XOR – matematický základ náhodnosti v PRNG.
""")
