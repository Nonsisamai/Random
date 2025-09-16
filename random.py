# mt_streamlit_break_predict.py
import streamlit as st
import time
import numpy as np
import plotly.express as px

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

# --------------------------
# Streamlit UI
# --------------------------
st.title("MTMini – prelomenie a 100 % presná predikcia")

st.markdown("""
- Generovanie, twist a temperovanie bitov.
- Vizualizácia rozptylu hodnôt (histogram + 3D scatter).
- Prelomenie generátora ukáže vnútorný stav, po ktorom môže byť predikcia presná.
""")

# Inicializácia
seed = st.number_input("Seed (celé číslo):", value=1, step=1)
if "mt" not in st.session_state or st.session_state.get("seed", None) != seed:
    st.session_state.mt = MTMini(seed)
    st.session_state.outputs = []
    st.session_state.seed = seed
    st.session_state.broken = False
    st.session_state.broken_state = None
    st.session_state.broken_index = None

mt = st.session_state.mt
outputs = st.session_state.outputs
broken = st.session_state.broken

speed = st.slider("Rýchlosť krokovania (sekundy na krok)", 0.1, 2.0, 0.5, 0.1)

# --------------------------
# Prelomenie generátora
# --------------------------
st.subheader("Prelomenie generátora")
if st.button("Break generator"):
    st.session_state.broken_state = mt.state.copy()
    st.session_state.broken_index = mt.index
    st.session_state.broken = True
    st.success("Generátor prelomený – vnútorný stav uložený.")
    st.text(f"State = {st.session_state.broken_state}")
    st.text(f"Index = {st.session_state.broken_index}")

# --------------------------
# Predikcia ďalšieho čísla
# --------------------------
st.subheader("Predikcia ďalšieho čísla")
if st.button("Predict next number"):
    if not broken:
        st.warning("Najprv je potrebné prelomiť generátor.")
    else:
        # Použijeme presný uložený stav generátora
        mt_temp = MTMini(seed)
        mt_temp.state = st.session_state.broken_state.copy()
        mt_temp.index = st.session_state.broken_index
        mt_temp.n = mt.n

        predicted_val = mt_temp.extract()
        st.success(f"Predikované číslo = {predicted_val} -> {bits(predicted_val)}")

        # Overenie: dočasná kópia pôvodného generátora
        mt_check = MTMini(seed)
        for val in outputs:
            mt_check.extract()
        next_val_actual = mt_check.extract()
        st.text(f"Skutočné číslo = {next_val_actual} -> {bits(next_val_actual)}")
        st.success(f"Predikcia {'sedí' if predicted_val == next_val_actual else 'nesedí'}!")

# --------------------------
# Generovanie čísla
# --------------------------
if st.button("Generate next number"):
    next_val = mt.extract()
    outputs.append(next_val)
    st.text(f"Generated number: {next_val} -> {bits(next_val)}")
    
    st.subheader("Detailný twist krok:")
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
# Vizualizácia
# --------------------------
if outputs:
    st.subheader("Histogram generovaných čísel")
    st.bar_chart(np.array(outputs))

    st.subheader("Bezpečný 3D scatter")
    df = np.array(outputs)
    if len(df) >= 3:
        n_points = len(df) // 3
        if n_points > 0:
            x = df[0:n_points]
            y = df[n_points:2*n_points]
            z = df[2*n_points:3*n_points]
        else:
            x = df
            y = np.zeros_like(x)
            z = np.zeros_like(x)
    else:
        x = df
        y = np.zeros_like(x)
        z = np.zeros_like(x)
    scatter_data = np.column_stack((x, y, z))
    fig = px.scatter_3d(
        x=scatter_data[:,0],
        y=scatter_data[:,1],
        z=scatter_data[:,2],
        color=scatter_data[:,0],
        labels={'x':'X','y':'Y','z':'Z'},
        title='Rozptyl pseudonáhodných hodnôt'
    )
    st.plotly_chart(fig)

# --------------------------
# Teória pseudonáhodnosti
# --------------------------
st.subheader("Teória pseudonáhodnosti a rozptylu")
st.markdown("""
- **Pseudonáhodnosť:** čísla vyzerajú náhodne, ale sú deterministické podľa seedu.
- **Rozptyl hodnôt:** vizualizuje, ako sa výstupy rozkladajú v priestore hodnôt.
- **Prelomenie:** získa aktuálny vnútorný stav generátora.
- **Predikcia:** po prelomení je 100 % presná.
- Každý krok twistu kombinuje horné a dolné bity, posúva ich a robí XOR – matematický základ pseudonáhodnosti.
""")
