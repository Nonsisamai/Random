# mt_streamlit_complete.py
import streamlit as st
import time
import numpy as np

# --------------------------
# Zjednodušený Mersenne Twister (pseudonáhodný)
# --------------------------
class MTMini:
    def __init__(self, seed, n=4, w=8, m=2):
        self.n = n          # počet stavových slov
        self.w = w          # počet bitov v slove
        self.m = m          # offset pri twist
        self.upper_mask = 0xF0
        self.lower_mask = 0x0F
        self.state = [(seed + i) % (1 << w) for i in range(n)]
        self.index = n      # núti twist pri prvom extracte

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
        # temperovanie
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
    # Inverzia temperovania (len z pedagogického dôvodu)
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
st.title("MTMini – pseudo-náhodnosť, vizualizácia a predikcia")

st.markdown("""
Tento program ukazuje, ako funguje pseudonáhodný generátor.  
- Generujeme čísla, ktoré **vyzerajú náhodne**.  
- Môžeme vizualizovať ich rozloženie **statisticky**.  
- Ukážeme **logiku deterministického predikovania** ďalšieho čísla, ak poznáme stav.
""")

seed = st.number_input("Seed (celé číslo):", value=1, step=1)
mt = MTMini(seed)

st.subheader("Počiatočný stav:")
for i, val in enumerate(mt.state):
    st.text(f"s[{i}] = {val} -> {bits(val)}")

# --------------------------
# Generovanie výstupov
# --------------------------
num_outputs = st.slider("Koľko výstupov generovať?", 3, 16, 6)
outputs = []
for i in range(num_outputs):
    outputs.append(mt.extract())

st.subheader("Generované čísla:")
st.text(outputs)

# --------------------------
# Statistická vizualizácia
# --------------------------
st.subheader("Vizualizácia rozloženia (statistická)")

# Histogram
st.bar_chart(np.array(outputs))

# Textová 3D ilúzia (len pedagogicky)
if len(outputs) >=3:
    st.subheader("Jednoduchá 3D vizualizácia náhodného priestoru")
    st.text(f"x={outputs[0]}, y={outputs[1]}, z={outputs[2]}")
    st.text("-> ukazuje, že čísla sú rozptýlené, vyzerajú náhodne")

# --------------------------
# Predikcia / prelomenie
# --------------------------
if st.button("Prelomiť a predpovedať ďalšie číslo"):
    st.subheader("Prelomenie PRNG a predikcia")
    # rekonštrukcia stavu z prvých n výstupov
    reconstructed_state = [invert_temper(val) for val in outputs[:mt.n]]
    st.text("Rekonštruovaný stav:")
    st.text(reconstructed_state)
    mt_reconstructed = MTMini(seed=0)
    mt_reconstructed.state = reconstructed_state
    mt_reconstructed.index = len(outputs) % mt_reconstructed.n
    next_val = mt_reconstructed.extract()
    st.text(f"Predpovedané ďalšie číslo = {next_val} -> {bits(next_val)}")
    st.markdown("""
    **Vysvetlenie:**  
    - PRNG je deterministický, takže ak poznáš stav, vieš predpovedať ďalšie číslo.  
    - Vyzerá náhodne, ale v skutočnosti je to "pseudonáhoda".  
    - Histogram a 3D text ukazujú, že čísla sú rovnomerne rozptýlené v priestore hodnôt.
    """)
