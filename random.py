# mt_streamlit_demo_animated.py
import streamlit as st
import time

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

def bits(x, w=8):
    return format(x, f'0{w}b')

# Streamlit UI
st.title("MTMini – animovaná vizualizácia bitov v twist kroku")

seed = st.number_input("Zadaj seed (celé číslo):", value=1, step=1)
mt = MTMini(seed)

st.subheader("Počiatočný stav:")
for i, val in enumerate(mt.state):
    st.text(f"s[{i}] = {val} -> {bits(val)}")

if st.button("Spusti animovaný twist"):
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
        time.sleep(1)  # pauza 1 sekunda pre animáciu

st.subheader("Generovanie výstupov:")
num_outputs = st.slider("Koľko výstupov generovať?", 1, 8, 4)
outputs = []
for i in range(num_outputs):
    val = mt.extract()
    outputs.append(val)
    st.text(f"Výstup {i}: {val} -> {bits(val)}")

if st.button("Prelomiť MT a predpovedať ďalšie číslo"):
    st.subheader("Rekonštrukcia a predikcia")
    def invert_temper(y):
        y_inv = y
        for _ in range(8):
            y_inv = y ^ ((y_inv << 1) & 0xB8)
        y_final = y_inv
        for _ in range(8):
            y_final = y_inv ^ (y_final >> 1)
        return y_final & 0xFF
    reconstructed_state = [invert_temper(val) for val in outputs[:mt.n]]
    for i, val in enumerate(reconstructed_state):
        st.text(f"s[{i}] rekonštruované = {val} -> {bits(val)}")
    mt_reconstructed = MTMini(seed=0)
    mt_reconstructed.state = reconstructed_state
    mt_reconstructed.index = len(outputs) % mt_reconstructed.n
    next_val = mt_reconstructed.extract()
    st.text(f"Predpovedané ďalšie číslo = {next_val} -> {bits(next_val)}")
