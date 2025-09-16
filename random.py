# mt_streamlit_final.py
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

    def twist_step(self, i, state=None):
        if state is None:
            state = self.state
        s_i = state[i]
        s_next = state[(i + 1) % self.n]
        upper = s_i & self.upper_mask
        lower = s_next & self.lower_mask
        y = upper | lower
        yA = y >> 1
        xor_const = 0xB8 if y & 1 else 0
        yA_final = yA ^ xor_const
        new_val = state[(i + self.m) % self.n] ^ yA_final
        return {"i": i, "s_i": s_i, "s_next": s_next, "upper": upper,
                "lower": lower, "y": y, "yA": yA, "xor_const": xor_const,
                "new_val": new_val}

    def twist_detailed(self, copy_state=None):
        # Ak sa da copy_state, vizualizujeme twist bez posunu pôvodného generátora
        if copy_state is None:
            copy_state = self.state.copy()
        new_state = []
        details = []
        for i in range(self.n):
            step = self.twist_step(i, copy_state)
            new_state.append(step["new_val"])
            details.append(step)
        return details, new_state

    def extract(self):
        if self.index >= self.n:
            _, self.state = self.twist_detailed()
            self.index = 0
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
st.title("MTMini – prelomenie a predikcia (presná a aproximovaná)")

st.markdown("""
- Generovanie, twist a temperovanie bitov.
- Vizualizácia rozptylu hodnôt (histogram + 3D scatter).
- Prelomenie generátora – presné aj aproximované.
- Predikcia ďalšieho čísla po prelomení.
""")

# Inicializácia
seed = st.number_input("Seed (celé číslo):", value=1, step=1)
if "mt" not in st.session_state or st.session_state.get("seed", None) != seed:
    st.session_state.mt = MTMini(seed)
    st.session_state.outputs = []
    st.session_state.seed = seed
    st.session_state.broken_exact = False
    st.session_state.broken_approx = False
    st.session_state.state_exact = None
    st.session_state.index_exact = None
    st.session_state.state_approx = None
    st.session_state.index_approx = None

mt = st.session_state.mt
outputs = st.session_state.outputs
speed = st.slider("Rýchlosť krokovania (sekundy na krok)", 0.1, 2.0, 0.5, 0.1)

# --------------------------
# Prelomenie generátora
# --------------------------
st.subheader("Prelomenie generátora")

# 1️⃣ Presné prelomenie
if st.button("Break generator (exact)"):
    st.session_state.state_exact = mt.state.copy()
    st.session_state.index_exact = mt.index
    st.session_state.broken_exact = True
    st.success("Generátor prelomený presne – vnútorný stav uložený.")
    st.text(f"State = {st.session_state.state_exact}")
    st.text(f"Index = {st.session_state.index_exact}")

# 2️⃣ Aproximované prelomenie (z temperovaných výstupov)
if st.button("Break generator (approx)"):
    if not outputs:
        st.warning("Najprv generuj aspoň jedno číslo.")
    else:
        # jednoduchá aproximácia: vezmeme posledné n výstupov ako stav
        st.session_state.state_approx = outputs[-mt.n:].copy() if len(outputs) >= mt.n else outputs.copy() + [0]*(mt.n - len(outputs))
        st.session_state.index_approx = 0
        st.session_state.broken_approx = True
        st.success("Generátor prelomený aproximovane.")
        st.text(f"Approximated State = {st.session_state.state_approx}")
        st.text(f"Index = {st.session_state.index_approx}")

# --------------------------
# Predikcia ďalšieho čísla
# --------------------------
st.subheader("Predikcia ďalšieho čísla")
pred_type = st.radio("Typ predikcie:", ("Presná", "Aproximovaná"))

if st.button("Predict next number"):
    if pred_type == "Presná" and not st.session_state.broken_exact:
        st.warning("Najprv je potrebné prelomiť generátor presne.")
    elif pred_type == "Aproximovaná" and not st.session_state.broken_approx:
        st.warning("Najprv je potrebné prelomiť generátor aproximovane.")
    else:
        if pred_type == "Presná":
            state_copy = st.session_state.state_exact.copy()
            index_copy = st.session_state.index_exact
        else:
            state_copy = st.session_state.state_approx.copy()
            index_copy = st.session_state.index_approx

        mt_temp = MTMini(seed)
        mt_temp.state = state_copy
        mt_temp.index = index_copy
        mt_temp.n = mt.n
        predicted_val = mt_temp.extract()
        st.success(f"Predikované číslo ({pred_type}) = {predicted_val} -> {bits(predicted_val)}")

        # Overenie: skutočné ďalšie číslo z generátora
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
    
    st.subheader("Detailný twist krok (vizualizácia, neposúva stav generátora):")
    details, _ = mt.twist_detailed(copy_state=mt.state.copy())
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

    st.subheader("3D scatter generovaných čísel")
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
- **Prelomenie presne:** uloží aktuálny vnútorný stav – predikcia 100 % presná.
- **Prelomenie aproximovane:** odhad vnútorného stavu z temperovaných výstupov – predikcia môže byť nepresná.
- Každý krok twistu kombinuje horné a dolné bity, posúva ich a robí XOR – matematický základ pseudonáhodnosti.
""")
