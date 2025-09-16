# mt_streamlit_full_demo_matplotlib.py
import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------
# MTMini - zjednodušený Mersenne Twister
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

def show_twist_step(step):
    st.text(f"Index {step['i']}:")
    st.text(f"s[i] = {step['s_i']} -> {bits(step['s_i'])}")
    st.text(f"s[i+1] = {step['s_next']} -> {bits(step['s_next'])}")
    st.text(f"upper = {step['upper']} -> {bits(step['upper'])}")
    st.text(f"lower = {step['lower']} -> {bits(step['lower'])}")
    st.text(f"y = upper|lower = {step['y']} -> {bits(step['y'])}")
    st.text(f"y >> 1 = {step['yA']} -> {bits(step['yA'])}")
    st.text(f"XOR const = {step['xor_const']} -> {bits(step['xor_const'])}")
    st.text(f"new state = {step['new_val']} -> {bits(step['new_val'])}")
    st.text("---")

# --------------------------
# Streamlit UI
# --------------------------
st.title("MTMini – vizuálna interaktívna náhoda a predikcia (Matplotlib)")

st.markdown("""
**Vysvetlenie "selsky":**  
Mersenne Twister mieša bity z počiatočného seedu.  
Každý krok twistu premieša horné a dolné bity, posúva ich a robí XOR.  
3D graf ukáže, ako sa hodnoty rozptýlia v priestore – vyzerá náhodne, ale je deterministické.
""")

# Inicializácia
seed = st.number_input("Zadaj seed (celé číslo):", value=1, step=1)
mt = MTMini(seed)

st.subheader("Počiatočný stav:")
for i, val in enumerate(mt.state):
    st.text(f"s[{i}] = {val} -> {bits(val)}")

# Animácia twistu
if st.button("Animovaný twist krok"):
    st.subheader("Animácia twistu bit po bite")
    placeholder = st.empty()
    details = mt.twist_detailed()
    for step in details:
        msg = f"Index {step['i']}:\n"
        msg += f"s[i] = {step['s_i']} -> {bits(step['s_i'])}\n"
        msg += f"s[i+1] = {step['s_next']} -> {bits(step['s_next'])}\n"
        msg += f"upper = {step['upper']} -> {bits(step['upper'])}\n"
        msg += f"lower = {step['lower']} -> {bits(step['lower'])}\n"
        msg += f"y = upper|lower = {step['y']} -> {bits(step['y'])}\n"
        msg += f"y >> 1 = {step['yA']} -> {bits(step['yA'])}\n"
        msg += f"XOR const = {step['xor_const']} -> {bits(step['xor_const'])}\n"
        msg += f"new state = {step['new_val']} -> {bits(step['new_val'])}\n"
        msg += "-------------------------"
        placeholder.text(msg)
        time.sleep(1)

# Generovanie výstupov
num_outputs = st.slider("Koľko výstupov generovať?", 1, 8, 4)
outputs = []
for i in range(num_outputs):
    val = mt.extract()
    outputs.append(val)
st.subheader("Výstupy:")
st.text(outputs)

# 3D scatter vizualizácia "náhodného priestoru" pomocou Matplotlib
if len(outputs) >= 3:
    st.subheader("3D vizualizácia náhodného priestoru")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(outputs[0], outputs[1], outputs[2], c='r', s=50)
    ax.set_xlabel('Output 0')
    ax.set_ylabel('Output 1')
    ax.set_zlabel('Output 2')
    st.pyplot(fig)

# Prelomenie a predikcia
if st.button("Prelomiť MT a vizualizovať predikciu"):
    st.subheader("Rekonštrukcia stavu a predikcia")
    reconstructed_state = [invert_temper(val) for val in outputs[:mt.n]]
    st.text("Rekonštruovaný stav:")
    st.text(reconstructed_state)

    mt_reconstructed = MTMini(seed=0)
    mt_reconstructed.state = reconstructed_state
    mt_reconstructed.index = len(outputs) % mt_reconstructed.n
    next_val = mt_reconstructed.extract()
    st.text(f"Predpovedané ďalšie číslo = {next_val} -> {bits(next_val)}")

    st.markdown("""
    **Vizualizácia pravdepodobnosti:**  
    Ak generujeme ďalšie čísla, vidíme, že MT je deterministický – ďalšie číslo sa presne zhoduje s predikciou.  
    3D scatter graf ukazuje, že hodnoty sa "rozptýlia" rovnomerne, takže vyzerá náhodne, ale je úplne predpovedateľné.
    """)
