import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import io
import zipfile
import base64
import itertools
import ast
from collections import Counter, defaultdict

# Pymatgen Imports
from pymatgen.core import Structure
from pymatgen.core.surface import Slab, SlabGenerator, get_symmetrically_distinct_miller_indices
from pymatgen.io.cif import CifWriter

# =============================================================================
# 1. VISUALIZATION ENGINE (NGL.js)
# =============================================================================

def visualize_with_ngl(structure, height=400, width="100%", unique_id="ngl_viewer"):
    """
    Embeds an interactive NGL.js 3D viewer into Streamlit using HTML components.
    """
    # 1. Convert Pymatgen Structure to CIF string
    cif_str = str(CifWriter(structure))
    
    # 2. Encode to Base64 to prevent special character issues in HTML/JS
    b64_str = base64.b64encode(cif_str.encode()).decode()

    # 3. Generate HTML Code
    # We use a unique ID for the viewport div to avoid conflicts if needed, 
    # though Streamlit iframes usually isolate them well.
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>NGL Viewer</title>
      <script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
      <style>
        body {{ margin: 0; padding: 0; overflow: hidden; }}
        #viewport_{unique_id} {{ width: 100%; height: {height}px; }}
      </style>
    </head>
    <body>
      <div id="viewport_{unique_id}"></div>
      <script>
        document.addEventListener("DOMContentLoaded", function () {{
          // Create NGL Stage
          var stage = new NGL.Stage("viewport_{unique_id}", {{backgroundColor: "white"}});
          
          // Decode Base64 CIF
          var stringBlob = new Blob([atob("{b64_str}")], {{type: 'text/plain'}});
          
          // Load and Render
          stage.loadFile(stringBlob, {{ext: "cif", defaultRepresentation: true}})
            .then(function (component) {{
              // Add representations
              component.addRepresentation("ball+stick", {{
                  aspectRatio: 1.5,
                  radiusScale: 0.3
              }});
              component.addRepresentation("unitcell");
              
              // Zoom to fit
              component.autoView();
            }});
            
          // Handle window resizing
          window.addEventListener("resize", function(event){{
              stage.handleResize();
          }}, true);
        }});
      </script>
    </body>
    </html>
    """
    
    # 4. Render component
    components.html(html_code, height=height)

# =============================================================================
# 2. CORE LOGIC (ChemicalAnalyzer & SlabProcessor)
# =============================================================================

class ChemicalAnalyzer:
    def __init__(self, structure):
        self.structure = structure
        self.centers = []       
        self.anions = []        
        self.cations = []       
        self.molecules = defaultdict(list) 
        self.covalent_map = {}  
        self.center_coordination = {}
        
        if len(self.structure) > 0:
            self._analyze()

    def _analyze(self):
        covalent_elements = set([
            'H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'I'
        ])
        
        adj_list = defaultdict(set)
        sites = self.structure.sites
        all_nn = self.structure.get_all_neighbors(r=2.5) 
        
        for i, neighbors in enumerate(all_nn):
            elem_i = sites[i].specie.element.symbol
            if elem_i not in covalent_elements: continue 
                
            for n in neighbors:
                elem_j = sites[n.index].specie.element.symbol
                if elem_j not in covalent_elements: continue
                
                dist = n.nn_distance
                is_bond = False
                
                # Dual Cutoff Logic
                if elem_i == 'H' or elem_j == 'H':
                    if dist < 1.25: is_bond = True
                else:
                    if dist < 2.3: is_bond = True
                
                if is_bond:
                    adj_list[i].add(n.index)
                    adj_list[n.index].add(i)

        visited = set()
        clusters = []
        
        for i in range(len(self.structure)):
            elem = sites[i].specie.element.symbol
            if elem not in covalent_elements:
                self.cations.append(i)
                visited.add(i)
                continue
            if i not in visited:
                component = []
                stack = [i]
                visited.add(i)
                while stack:
                    curr = stack.pop()
                    component.append(curr)
                    for neighbor in adj_list[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
                clusters.append(component)

        try:
            unique_species = sorted(list(set([s.specie for s in self.structure])), key=lambda x: x.oxi_state)
            candidates = [s for s in unique_species if s.element.symbol not in ['H', 'O'] and s.oxi_state > 0]
            center_specie_symbol = candidates[-1].element.symbol if candidates else None
        except:
            center_specie_symbol = None

        for cluster in clusters:
            elements = [sites[x].specie.element.symbol for x in cluster]
            is_scaffold = False
            if center_specie_symbol and center_specie_symbol in elements:
                is_scaffold = True
            
            if is_scaffold:
                c_indices = [x for x in cluster if sites[x].specie.element.symbol == center_specie_symbol]
                for x in cluster:
                    if x in c_indices:
                        self.centers.append(x)
                        self.covalent_map[x] = cluster
                        anions_in_unit = [a for a in cluster if sites[a].specie.element.symbol == 'O']
                        self.center_coordination[x] = len(anions_in_unit)
                    else:
                        self.anions.append(x)
            else:
                counts = Counter(elements)
                formula = "".join(sorted([f"{k}{v if v>1 else ''}" for k,v in counts.items()]))
                self.molecules[formula].append(cluster)

class SlabProcessor:
    def __init__(self, slab, bulk_analyzer):
        self.slab = slab
        self.bulk_analyzer = bulk_analyzer
        self.local_analysis = ChemicalAnalyzer(self.slab)
        
        n_bulk_centers = len(bulk_analyzer.centers)
        if n_bulk_centers == 0: 
            self.ratio_cation = 0
            self.ratio_mols = {}
        else:
            self.ratio_cation = len(bulk_analyzer.cations) / n_bulk_centers
            self.ratio_mols = {}
            for formula, units in bulk_analyzer.molecules.items():
                self.ratio_mols[formula] = len(units) / n_bulk_centers

    def remove_indices(self, indices_to_remove):
        if not indices_to_remove: return
        self.slab.remove_sites(list(indices_to_remove))
        if len(self.slab) > 0:
            self.local_analysis = ChemicalAnalyzer(self.slab)
        else:
            self.local_analysis = None

    def step_5_strip_to_scaffold(self):
        to_remove = set()
        for mol_list in self.local_analysis.molecules.values():
            for unit in mol_list: to_remove.update(unit)
        to_remove.update(self.local_analysis.cations)
        
        if self.bulk_analyzer.centers:
            bulk_center_idx = self.bulk_analyzer.centers[0]
            target_cn = self.bulk_analyzer.center_coordination.get(bulk_center_idx, 4)
            for c_idx, u_indices in self.local_analysis.covalent_map.items():
                current_cn = self.local_analysis.center_coordination.get(c_idx, 0)
                if current_cn < target_cn: to_remove.update(u_indices)
        
        valid = set(self.local_analysis.cations)
        for u in self.local_analysis.covalent_map.values(): valid.update(u)
        for mlist in self.local_analysis.molecules.values():
            for u in mlist: valid.update(u)
            
        for i in range(len(self.slab)):
            if i not in valid: to_remove.add(i)

        self.remove_indices(to_remove)

    def step_6_depolarize_scaffold(self):
        if not self.local_analysis or not self.local_analysis.covalent_map: return False

        units = []
        for c, u in self.local_analysis.covalent_map.items():
            units.append((self.slab[c].frac_coords[2], u))
        units.sort(key=lambda x: x[0])
        
        max_rem = len(units) // 2
        for n_top in range(max_rem):
            for n_bot in range(max_rem):
                if n_top + n_bot >= len(units): continue
                test = self.slab.copy()
                to_del = []
                if n_bot: 
                    for i in range(n_bot): to_del.extend(units[i][1])
                if n_top: 
                    for i in range(n_top): to_del.extend(units[-(i+1)][1])
                
                if to_del: test.remove_sites(to_del)
                if len(test) == 0: continue
                if not test.is_polar():
                    self.slab = test
                    self.local_analysis = ChemicalAnalyzer(self.slab)
                    return True
        return False

def structure_to_slab(structure, ref_slab):
    return Slab(structure.lattice, structure.species, structure.frac_coords,
                ref_slab.miller_index, ref_slab.oriented_unit_cell,
                ref_slab.shift, ref_slab.scale_factor, site_properties=structure.site_properties)

# =============================================================================
# 3. STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Complex Mineral Surface Generator", layout="wide")

st.title("💎 Complex Mineral Surface Generator")
st.markdown("""
This tool generates non-polar, stoichiometric surfaces for complex minerals. 
It now includes **Interactive 3D Visualization** via NGL.js.
""")

# --- SIDEBAR: PARAMETERS ---
with st.sidebar:
    st.header("1. Structure Input")
    uploaded_file = st.file_uploader("Upload Bulk CIF", type=["cif"])
    
    st.header("2. Cut Selection Mode")
    mode = st.radio("Miller Index Selection:", 
                    ["Single Index", "Up to Max Index", "Specific List"])
    
    target_indices = []
    
    if mode == "Single Index":
        col_h, col_k, col_l = st.columns(3)
        h = col_h.number_input("h", value=1, step=1)
        k = col_k.number_input("k", value=0, step=1)
        l = col_l.number_input("l", value=0, step=1)
        target_indices = [(h, k, l)]
        
    elif mode == "Up to Max Index":
        max_h = st.number_input("Max Miller Index", value=1, min_value=1, max_value=5, step=1)
        st.info("Indices will be calculated based on bulk symmetry.")
        
    elif mode == "Specific List":
        st.caption("Enter indices as tuples, e.g., `(1,0,0), (0,1,1)`")
        indices_str = st.text_area("List of Indices", value="(1,0,0)")

    st.header("3. Slab Settings")
    thickness = st.number_input("Min Thickness (Å)", value=15.0, step=1.0)
    vacuum = st.number_input("Vacuum Size (Å)", value=15.0, step=1.0)
    
    generate_btn = st.button("Generate Surfaces", type="primary")

# --- MAIN EXECUTION ---
if uploaded_file and generate_btn:
    # 1. Load Structure
    try:
        cif_string = uploaded_file.getvalue().decode("utf-8")
        bulk = Structure.from_str(cif_string, fmt="cif")
        
        try: bulk.add_oxidation_state_by_guess()
        except: bulk.add_oxidation_state_by_guess(max_sites=-1)
        
        st.success(f"Loaded {uploaded_file.name} | Formula: {bulk.formula}")
        
    except Exception as e:
        st.error(f"Error reading CIF: {e}")
        st.stop()

    # 2. Determine Indices
    if mode == "Up to Max Index":
        with st.spinner(f"Calculating symmetric indices up to max index {max_h}..."):
            target_indices = get_symmetrically_distinct_miller_indices(bulk, max_h)
            st.write(f"**Found {len(target_indices)} distinct indices:** {target_indices}")
            
    elif mode == "Specific List":
        try:
            parsed = ast.literal_eval(f"[{indices_str}]") if not indices_str.startswith('[') else ast.literal_eval(indices_str)
            target_indices = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    target_indices.append(tuple(item))
            if not target_indices: raise ValueError
        except:
            st.error("Invalid format for index list.")
            st.stop()

    # 3. Pre-Analysis
    with st.status("Analyzing Bulk Chemistry...", expanded=True) as status:
        bulk_an = ChemicalAnalyzer(bulk)
        
        col1, col2 = st.columns(2)
        with col1:
            if bulk_an.centers:
                center_el = bulk.sites[bulk_an.centers[0]].specie.element.symbol
                st.metric("Scaffold Center", f"{center_el} ({len(bulk_an.centers)} sites)")
            else:
                st.error("No covalent scaffold found!")
                st.stop()
        
        with col2:
            st.metric("Counter-Cations", len(bulk_an.cations))
        
        if bulk_an.molecules:
            st.write("**Intercalated Molecules:**")
            st.json({k: len(v) for k,v in bulk_an.molecules.items()})
        else:
            st.info("No neutral molecules detected.")

        status.update(label="Bulk Analysis Complete", state="complete", expanded=False)

    # 4. Generation Loop
    progress_bar = st.progress(0)
    status_text = st.empty()
    final_slabs = []
    
    total_indices = len(target_indices)
    
    for idx_i, current_hkl in enumerate(target_indices):
        status_text.text(f"Processing Index {current_hkl} ({idx_i+1}/{total_indices})...")
        progress_bar.progress((idx_i) / total_indices)
        
        try:
            slabgen = SlabGenerator(bulk, current_hkl, thickness, vacuum, center_slab=True)
            raw_slabs = slabgen.get_slabs()
            
            for raw_slab in raw_slabs:
                proc_slab = raw_slab.get_orthogonal_c_slab()
                raw_an = ChemicalAnalyzer(proc_slab)
                
                cat_cands = [proc_slab[x] for x in raw_an.cations]
                mol_cands_map = {}
                for form, units in raw_an.molecules.items():
                    mol_cands_map[form] = []
                    for u_indices in units:
                        mol_cands_map[form].append([proc_slab[x] for x in u_indices])
                
                proc = SlabProcessor(proc_slab.copy(), bulk_an)
                proc.step_5_strip_to_scaffold()
                
                if not proc.local_analysis or not proc.local_analysis.centers: continue 
                
                if not proc.step_6_depolarize_scaffold(): continue 
                
                scaffold = proc.slab
                center_z = 0.5
                n_centers = len(proc.local_analysis.centers)
                req_cats = int(round(n_centers * proc.ratio_cation))
                cat_cands.sort(key=lambda s: abs(s.frac_coords[2] - center_z))
                
                if len(cat_cands) < req_cats: continue
                
                best_base = None
                current_base = [s for s in scaffold]
                base_cat = cat_cands[:max(0, req_cats-2)]
                pool_cat = cat_cands[max(0, req_cats-2) : min(len(cat_cands), req_cats+4)]
                needed = req_cats - len(base_cat)
                
                lowest_dip = float('inf')
                
                for combo in itertools.combinations(pool_cat, needed):
                    test_sites = current_base + base_cat + list(combo)
                    ts_struct = Structure(scaffold.lattice, [s.specie for s in test_sites], [s.frac_coords for s in test_sites])
                    ts_slab = structure_to_slab(ts_struct, proc_slab)
                    
                    dip = abs(ts_slab.dipole[2])
                    if dip < lowest_dip: lowest_dip = dip
                    
                    if not ts_slab.is_polar():
                        best_base = ts_slab
                        break
                
                if not best_base: continue
                
                final_sites = [s for s in best_base]
                success_mols = True
                for form, ratio in proc.ratio_mols.items():
                    req_n = int(round(n_centers * ratio))
                    if req_n == 0: continue
                    cands = mol_cands_map.get(form, [])
                    cands.sort(key=lambda unit: abs(unit[0].frac_coords[2] - center_z))
                    if len(cands) < req_n:
                        success_mols = False
                        break
                    for unit in cands[:req_n]: final_sites.extend(unit)
                
                if not success_mols: continue
                
                final_struct = Structure(scaffold.lattice, [s.specie for s in final_sites], [s.frac_coords for s in final_sites])
                final_slab = structure_to_slab(final_struct, proc_slab)
                final_slabs.append(final_slab)

        except Exception as e:
            st.warning(f"Calculation failed for index {current_hkl}: {e}")
            continue

    progress_bar.progress(1.0)
    status_text.text("Generation Complete!")

    # --- RESULTS DISPLAY ---
    st.header(f"Results: {len(final_slabs)} Valid Surfaces Found")
    
    if len(final_slabs) == 0:
        st.warning("No valid non-polar, stoichiometric surfaces found.")
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for i, slab in enumerate(final_slabs):
            
            an = ChemicalAnalyzer(slab)
            mult = len(slab) / len(bulk)
            dipole_z = slab.dipole[2]
            is_sym = slab.is_symmetric()
            thickness_val = np.ptp(slab.cart_coords[:, 2])
            
            hkl_str = "".join(map(str, slab.miller_index))
            fname = f"slab_{hkl_str}_id{i}.cif"
            
            w = CifWriter(slab)
            zip_file.writestr(fname, str(w))
            
            # === UI CARD WITH NGL ===
            with st.expander(f"Surface ({hkl_str}) | ID: {i} | Polarity: {dipole_z:.6f} eÅ", expanded=False):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### Metadata")
                    st.write(f"**Filename:** `{fname}`")
                    st.write(f"**Multiplicity:** {mult:.2f}x Bulk")
                    st.write(f"**Thickness:** {thickness_val:.3f} Å")
                    st.write(f"**Symmetry:** {is_sym}")
                    st.write(f"**Dipole Z:** {dipole_z:.6f}")
                    st.download_button("Download CIF", str(w), fname, "chemical/x-cif", key=f"dl_{i}")

                with col2:
                    st.markdown("### 3D Visualization")
                    # CALLING THE NGL VISUALIZER HERE
                    visualize_with_ngl(slab, unique_id=f"ngl_{i}")

    if final_slabs:
        st.download_button(
            label="Download All Surfaces (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"surfaces_results.zip",
            mime="application/zip",
            type="primary"
        )
