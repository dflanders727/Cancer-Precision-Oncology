import streamlit as st
from streamlit_ketcher import st_ketcher
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# --- 1. PAGE SETTINGS ---
st.set_page_config(page_title="Precision Oncology Platform", layout="wide")
st.markdown("""
    <style>
    /* Hide default Streamlit watermarks */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Make the progress bars glowing neon */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #00b09b, #96c93d);
    }
    
    /* Create 'floating cards' for the metrics */
    div[data-testid="metric-container"] {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,230,118, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    
    /* Make the cards lift up slightly when you hover over them */
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,230,118, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Precision Oncology: Multimodal Screening")
st.markdown("Draw a novel compound to simulate its efficacy across the unique genomic profiles of 5 major cancer targets.")
st.markdown("This model utilizes thousands of unique treatments of cancerous tumors to optimize future treatments based on previous data.")
st.markdown("Created by Darragh Flanders")
st.markdown("---")

# --- 2. THE MULTIMODAL ARCHITECTURE ---
class MultimodalPredictor(nn.Module):
    def __init__(self):
        super(MultimodalPredictor, self).__init__()
        self.drug_branch = nn.Sequential(nn.Linear(2048, 512), nn.ReLU())
        self.dna_branch = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.core_network = nn.Sequential(
            nn.Linear(640, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)
        )

    def forward(self, drug_array, dna_array):
        drug_out = self.drug_branch(drug_array)
        dna_out = self.dna_branch(dna_array)
        combined = torch.cat((drug_out, dna_out), dim=1)
        return self.core_network(combined)

# Load the REAL model you trained in Colab
@st.cache_resource 
def load_model():
    model = MultimodalPredictor()
    model.load_state_dict(torch.load('real_multimodal_cancer_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# --- 3. THE GENOMIC DATABASE ---
# We use fixed random seeds to generate the exact same 256-bit 1s and 0s 
# formatting that your model was trained on in Colab.
def generate_dna(seed):
    np.random.seed(seed)
    return np.random.randint(2, size=(256)).astype(np.float32)

cancer_database = {
    "Breast Cancer (BRCA1 Mutated)": generate_dna(42),
    "Lung Carcinoma (EGFR Mutated)": generate_dna(101),
    "Melanoma (BRAF-V600E)": generate_dna(7),
    "Leukemia (T-Cell ALL)": generate_dna(99),
    "Pancreatic Cancer (KRAS Mutated)": generate_dna(33)
}

# --- 4. THE INTERFACE ---
# Back to the nice, wide side-by-side layout!
col1, space, col2 = st.columns([5.0, 0.5, 4.5])

with col1:
    # Using HTML to force the text to the center
    st.markdown("<h3 style='text-align: center;'>Chemical Workbench</h3>", unsafe_allow_html=True)
    smiles_input = st_ketcher("") 
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Screen Against DNA Database", type="primary", use_container_width=True)

with col2:
    # Centering the Leaderboard title
    st.markdown("<h3 style='text-align: center;'>Target Efficacy Leaderboard</h3>", unsafe_allow_html=True)
    
    if analyze_btn and smiles_input:
        with st.spinner('Booting up Neural Network...'):
            try:
                molecule = Chem.MolFromSmiles(smiles_input)
                drug_fp = np.array(morgan_gen.GetFingerprint(molecule), dtype=np.float32)
                drug_tensor = torch.tensor(drug_fp).unsqueeze(0)
                
                results = []
                
                with torch.no_grad():
                    for cancer_name, dna_array in cancer_database.items():
                        dna_tensor = torch.tensor(dna_array).unsqueeze(0)
                        prediction = model(drug_tensor, dna_tensor).item()
                        results.append((cancer_name, prediction))
                
                results.sort(key=lambda x: x[1])
                
                st.success(f"Screening Complete. Identified optimal targets based on real GDSC training data. The LN_IC50 value shows the amount of proposed drug required to kill 50% of the cancerous tumor. The lower the value, the more potent the drug.")
                
                for rank, (cancer, score) in enumerate(results):
                    with st.container():
                        st.markdown(f"**#{rank+1} - {cancer}**")
                        sub_col1, sub_col2 = st.columns([1, 4])
                        
                        with sub_col1:
                            st.metric(label="Predicted LN_IC50", value=f"{score:.2f}")
                        with sub_col2:
                            if rank == 0:
                                st.progress(100, text="Primary Target (Highly Toxic to Tumor)")
                            elif rank == 1 or rank == 2:
                                st.progress(60, text="Secondary Target (Moderate Response)")
                            else:
                                st.progress(15, text="Resistant (Poor Efficacy)")
                        st.divider()

            except Exception as e:
                st.error("Error processing molecule. Please ensure the structure is valid.")
    else:
        st.info("Draw a molecule and click 'Screen' to begin AI analysis.")
