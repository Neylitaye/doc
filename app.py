# -*- coding: utf-8 -*-
import os, io, time, datetime, tempfile, base64, re, json, hashlib
import requests
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import fitz  # PyMuPDF

import torch
import torch.nn as nn
from torchvision import models, transforms

from ultralytics import YOLO
import easyocr

# =========================
# CONFIG GLOBALE
# =========================
st.set_page_config(page_title="Analyse IA - Docs administratifs", layout="wide")
st.markdown("<style>footer{visibility:hidden;}</style>", unsafe_allow_html=True)

# --- Chemins des modèles (adapte si besoin)
YOLO_PATH = "yolov8_doc_elements.pt"                # modèle de détection
CNN_PATH  = "efficientnet_doc_classifier_best.pth"  # modèle de classification

# --- Dossiers de sortie
BASE_DIR = "documents_enregistres"
os.makedirs(BASE_DIR, exist_ok=True)

# --- Domain guard (Assistant IA)
DOMAIN_KEYWORDS = [
    "arrêté","arrete","décret","decret","circulaire","note de service","cachet","emblème","devise",
    "timbre","république du cameroun","ministère","ministre","préfet","gouverneur","communiqué",
    "signature","numéro","référence","scellé","acte","document administratif"
]

# --- Styles UI
st.markdown("""
<style>
:root { --brand:#4A90E2; }
.block-container { padding-top: 1rem !important; }
h1,h2,h3 { font-weight: 800; }
.chat-panel {
  width: 100%; max-width: 820px; margin: 0 auto; background:#121212; color:#e5e5e5;
  border:1px solid #2a2a2a; border-radius: 16px; box-shadow:0 8px 24px rgba(0,0,0,.3);
}
.chat-header { padding: 12px 16px; display:flex; gap:8px; align-items:center; border-bottom: 1px solid #2a2a2a; }
.chat-title { font-weight:700; margin:0; }
.chat-body { padding: 12px; max-height: 52vh; overflow-y: auto; }
.msg-user,.msg-bot { padding:10px 12px; border-radius:12px; margin-bottom:8px; width:fit-content; max-width:92%;
  white-space:pre-wrap; word-wrap:break-word; }
.msg-user { background:#243447; margin-left:auto; }
.msg-bot { background:#1f2937; }
</style>
""", unsafe_allow_html=True)

# =========================
# STATES
# =========================
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# =========================
# RESSOURCES (modèles)
# =========================
@st.cache_resource(show_spinner=True)
def load_yolo():
    return YOLO(YOLO_PATH)

@st.cache_resource(show_spinner=True)
def load_cnn(num_classes=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None)
    num_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_f, num_classes)
    state = torch.load(CNN_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model, device

@st.cache_resource(show_spinner=True)
def load_easyocr():
    # GPU=False pour éviter les plantages CUDA sur certaines configs
    return easyocr.Reader(['fr','en'], gpu=False)

# Chargement avec gestion d'erreurs
yolo_model = None
cnn_model, device = None, None
reader = None

try:
    yolo_model = load_yolo()
except Exception as e:
    st.warning(f"⚠️ YOLO non chargé : {e}")

try:
    cnn_model, device = load_cnn(num_classes=3)  # adapte mapping des classes plus bas
except Exception as e:
    st.warning(f"⚠️ CNN non chargé : {e}")

try:
    reader = load_easyocr()
except Exception as e:
    st.warning(f"⚠️ EasyOCR non chargé : {e}")

# =========================
# TRANSFORMS + UTILS
# =========================
transform_224 = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

ID2LABEL = {0:"arrete", 1:"circulaire", 2:"decret"}  # adapte à ton entraînement

def is_in_domain(q:str)->bool:
    ql = q.lower()
    return any(kw in ql for kw in DOMAIN_KEYWORDS)

def pdf_to_images(pdf_bytes: bytes):
    """Convertit PDF en liste d'images PIL."""
    images = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes); tmp.flush()
        doc = fitz.open(tmp.name)
        for p in doc:
            pix = p.get_pixmap()
            img_bytes = pix.tobytes("png")
            images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        doc.close()
    return images

def run_yolo_pil(pil_img: Image.Image):
    if yolo_model is None:
        return None
    arr = np.array(pil_img)
    return yolo_model(arr)

def classify_pil(pil_img: Image.Image):
    if cnn_model is None: return "inconnu"
    with torch.no_grad():
        t = transform_224(pil_img).unsqueeze(0).to(device)
        out = cnn_model(t)
        pred = out.argmax(1).item()
        return ID2LABEL.get(pred, f"classe_{pred}")

def extract_ocr_text(pil_img: Image.Image):
    if reader is None: return ""
    arr = np.array(pil_img)  # EasyOCR accepte np.ndarray (RGB)
    lines = reader.readtext(arr, detail=0)
    return "\n".join(lines)

# ======= Noms sûrs et sauvegarde anti-doublons =======

def safe_filename(name: str) -> str:
    """Nettoie une chaîne pour en faire un nom de fichier valide (Windows/Linux/Mac)."""
    name = name.replace("\n", " ").replace("\r", " ").strip()
    name = re.sub(r'[\\/*?:"<>|]', "_", name)  # remplacer caractères interdits
    name = re.sub(r"\s+", " ", name)
    return name[:100] if len(name) > 100 else name

def sha1_text(txt: str) -> str:
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()

def _load_index(folder: str) -> dict:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "_index_sha1.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def _save_index(folder: str, idx: dict):
    path = os.path.join(folder, "_index_sha1.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

def save_document(doc_class: str,
                  file_name: str,
                  file_bytes: bytes,
                  extracted_text,
                  preview_image: Image.Image | None = None,
                  title_hint: str | None = None):
    """
    Sauvegarde un document administratif :
    - Texte OCR dans un .txt (fusionné si PDF multi-pages)
    - Image d’aperçu en .png (1ère page si PDF, sinon image fournie)
    - Anti-doublons par hash SHA-1 du texte
    - Noms fichier: <classe>_<YYYYMMDD_HHMMSS>.{txt|png}

    Returns:
        dict: {"txt": chemin_txt, "img": chemin_img, "created": bool}
    """
    # Dossier de la classe
    folder = os.path.join(BASE_DIR, doc_class)
    os.makedirs(folder, exist_ok=True)

    # Texte final (fusion pour listes)
    if isinstance(extracted_text, list):
        text_final = "\n\n".join(extracted_text)
    else:
        text_final = extracted_text or ""

    # Anti-doublon via hash du texte
    idx = _load_index(folder)
    h = sha1_text(text_final)
    if h in idx:
        # Déjà enregistré
        existing = idx[h]
        return {
            "txt": os.path.join(folder, existing["txt"]),
            "img": os.path.join(folder, existing["img"]) if existing.get("img") else None,
            "created": False
        }

    # Base name (timestamp sûr)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{doc_class}_{ts}"
    base_name = safe_filename(base_name)

    # 1) Sauvegarde texte
    txt_name = base_name + ".txt"
    txt_path = os.path.join(folder, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_final)

    # 2) Sauvegarde image d’aperçu
    img_name = base_name + ".png"
    img_path = os.path.join(folder, img_name)
    try:
        if file_name.lower().endswith(".pdf"):
            # Preview à partir des bytes PDF (1ère page)
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(pdf_doc) > 0:
                page = pdf_doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                img.save(img_path, "PNG")
            else:
                img_path = None
        else:
            # Image directe si fournie
            if preview_image is not None:
                preview_image.convert("RGB").save(img_path, "PNG")
            else:
                img_path = None
    except Exception as e:
        print("⚠️ Erreur création preview image:", e)
        img_path = None

    # Mettre à jour l'index anti-doublon
    idx[h] = {"txt": txt_name, "img": img_name if img_path else None}
    _save_index(folder, idx)

    return {"txt": txt_path, "img": img_path, "created": True}

# =========================
# HUGGING FACE CHAT (Phi 3.5)
# =========================
API_URL_CHAT = "https://router.huggingface.co/hf-inference/models/microsoft/Phi-3.5-mini-instruct/v1/chat/completions"

def query_chat(payload, api_key):
    headers = { "Authorization": f"Bearer {api_key}" }
    r = requests.post(API_URL_CHAT, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


# =========================
# TABS
# =========================
tab_accueil, tab_analyse, tab_stats, tab_bd, tab_assistant = st.tabs(
    ["🏠 Accueil", "🧪 Analyse", "📊 Statistiques", "📂 Bd", "🤖 Chatbot"]
)

# -------------------------
# ACCUEIL
# -------------------------
with tab_accueil:
    st.markdown("<h1 style='text-align:center;'>DocuVision IA</h1>", unsafe_allow_html=True)
    text = "Bienvenue sur l'application de reconnaissance intelligente de documents administratifs au Cameroun ."
    placeholder = st.empty()
    displayed = ""
    for ch in text:
        displayed += ch
        placeholder.markdown(f"<h3 style='text-align:center;color:#4A90E2'>{displayed}</h3>", unsafe_allow_html=True)
        time.sleep(0.01)
    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "### 📄 Définition du document administratif""\n"
            "Le document administratif est défini comme tout support écrit, visuel ou numérique produit, "
            "reçu ou conservé par une entité publique ou parapublique dans le cadre de ses missions de service public "
            "(Ministère de la Fonction Publique du Cameroun, 2022).\n\n"
            "Il peut s’agir d’un acte, d’un formulaire, d’un rapport ou d’une correspondance officielle, "
            "servant à formaliser une décision, attester d’un droit ou faciliter une procédure administrative.\n\n"
            "Ces documents sont essentiels à la traçabilité, à la transparence et à la légalité des actions publiques.\n\n"
            "Au Cameroun, les documents administratifs sont régis par des textes comme le *Décret n° 2008/035 du 23 janvier 2008* "
            "portant organisation des archives publiques, qui impose leur conservation et leur accessibilité selon des normes précises. "
            "Ils jouent un rôle crucial dans les interactions entre citoyens et institutions.), notamment pour :" "\n"
            "- l’obtention de prestations sociales""\n"
            "- la régularisation de situations juridiques""\n"
            "- la gestion des carrières dans la fonction publique"
        )

    with col2:
        st.info(
            "### 🧩 Éléments constitutifs d’un document administratif" "\n"
            "- Cachet""\n"
            "- Emblème national""\n"
            "- Signature""\n"
            "- Devise""\n"
            "- Titre"
        )
        col3, col4 = st.columns(2)
        with col3:
            st.image("https://thaka.bing.com/th/id/OIP.by5yaBY_V4ueHiDVNWxPXAHaFP?w=266&h=188&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3", width=300)
        with col4:
            st.image("https://tse4.mm.bing.net/th/id/OIP.Nkqj9LsSzH9_DgH4Lf6RBwHaE0?rs=1&pid=ImgDetMain&o=7&rm=3", width=350)
    st.success("Lance-toi dans l'onglet 'Analyse' pour tester la reconnaissance de documents !")

# -------------------------
# ANALYSE
# -------------------------
with tab_analyse:
    st.header("Analyse de documents")
    uploaded = st.file_uploader("Importer un PDF ou une image (PNG/JPG)", type=["pdf","png","jpg","jpeg"])
    run = st.button("Lancer l'analyse")

    if uploaded and run:
        try:
            # IMPORTANT : lire les bytes UNE SEULE FOIS
            file_bytes = uploaded.read()
            file_name = uploaded.name

            # Déterminer si PDF
            is_pdf = file_name.lower().endswith(".pdf")

            if is_pdf:
                images = pdf_to_images(file_bytes)
            else:
                images = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

            yolo_summaries = []
            any_detection = False
            ocr_pages = []  # Pour fusionner l'OCR des PDF

            # On affiche / détecte / OCR pour chaque page/image
            for i, im in enumerate(images[:10]):  # limite 10 pages pour éviter les abus
                st.subheader(f"Page/Image {i+1}")

                # YOLO
                if yolo_model:
                    res = yolo_model(np.array(im))
                    plot = res[0].plot()  # image annotée (numpy BGR)
                    st.image(plot, caption="Détections YOLO", use_column_width=True)
                    boxes = res[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        names = res[0].names
                        lbls = [names[int(b.cls[0])] for b in boxes]
                        yolo_summaries.append(", ".join(lbls))
                        any_detection = True
                    else:
                        yolo_summaries.append("aucun élément détecté")
                else:
                    st.info("YOLO non disponible")

                # CNN (classification)
                label = classify_pil(im)
                st.success(f"Classification CNN : **{label}**")

                # OCR
                txt = extract_ocr_text(im)
                if txt.strip():
                    st.text_area(f"Texte OCR extrait — Page {i+1}", txt, height=160)
                else:
                    st.warning("Aucun texte OCR détecté ou EasyOCR indisponible.")
                ocr_pages.append(txt)

            # ---- SAUVEGARDE (une seule fois) ----
            # Classe prioritaire : si plusieurs pages classées différemment, on prend la plus fréquente
            if images:
                # recalcul de la classe majoritaire sur l'ensemble
                classes = []
                for im in images:
                    classes.append(classify_pil(im))
                # filtrer pour valeurs valides
                classes = [c for c in classes if c in ID2LABEL.values()]
                final_class = classes[0] if not classes else max(set(classes), key=classes.count)
            else:
                final_class = "inconnu"

            # Image preview :
            preview_image = images[0] if images else None

            # Sauvegarder seulement si texte non vide et classe valide
            if final_class in ID2LABEL.values() and any(t.strip() for t in ocr_pages):
                res = save_document(
                    doc_class=final_class,
                    file_name=file_name,
                    file_bytes=file_bytes,
                    extracted_text=ocr_pages if is_pdf else (ocr_pages[0] if ocr_pages else ""),
                    preview_image=preview_image
                )
                if res.get("created", False):
                    st.toast("Document sauvegardé ✅", icon="✅")
                else:
                    st.toast("Document déjà enregistré ✅", icon="✅")
            else:
                st.info("🔎 Rien à sauvegarder (classe inconnue ou aucun texte OCR détecté).")

            if yolo_summaries:
                st.info("Synthèse YOLO : " + " | ".join(yolo_summaries))
            if not any_detection:
                st.warning("Aucune détection administrative nette. Vérifie la qualité du scan.")

        except Exception as e:
            st.error(f"Erreur d'analyse : {e}")

# -------------------------
# STATISTIQUES
# -------------------------
with tab_stats:
    st.header("📊 Statistiques des documents enregistrés")
    
    class_dirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

    counts = {}
    timeline = []  # (date, class)
    for c in class_dirs:
        p = os.path.join(BASE_DIR, c)
        files = [f for f in os.listdir(p) if f.endswith(".txt")]
        counts[c] = len(files)
        for f in files:
            try:
                base = os.path.splitext(f)[0]
                ts = "_".join(base.split("_")[-2:])
                dt = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S")
            except:
                dt = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(p, f)))
            timeline.append((dt.date(), c))

    if not counts:
        st.info("Aucun fichier enregistré pour le moment. Lance d'abord des analyses.")
    else:
        import pandas as pd

        # Créer toutes les figures dans une liste
        figs = []

        # ---- Camembert ----
        fig1, ax1 = plt.subplots(figsize=(5,4))
        labels = list(counts.keys())
        sizes  = [counts[k] for k in labels]
        ax1.pie(
            sizes,
            labels=labels,
            autopct=lambda pct: f"{int(round(pct/100.*sum(sizes)))}",
            startangle=90,
            textprops={'fontsize': 10}
        )
        ax1.axis('equal')
        figs.append(fig1)

        # ---- Courbes cumulatives ----
        if timeline:
            df = pd.DataFrame(timeline, columns=["date", "classe"])
            df_count = df.groupby(["date", "classe"]).size().unstack(fill_value=0)
            df_cumu = df_count.cumsum()

            for col_name in df_cumu.columns:
                fig, ax = plt.subplots(figsize=(5,4))
                ax.plot(df_cumu.index, df_cumu[col_name], linewidth=2, marker="o")
                ax.set_xlabel("Date")
                ax.set_ylabel("Nombre cumulé")
                ax.set_title(f"Évolution - {col_name}")
                ax.grid(True, alpha=.3)
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d/%m/%Y"))
                fig.autofmt_xdate(rotation=45)
                figs.append(fig)

        # ---- Affichage flottant 2 par ligne ----
        for i in range(0, len(figs), 2):
            cols = st.columns(2)
            cols[0].pyplot(figs[i])
            if i+1 < len(figs):
                cols[1].pyplot(figs[i+1])


# -------------------------
# Base de données
# -------------------------
with tab_bd:
    st.header("📂 Base de données des documents enregistrés")

    class_dirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

    all_docs = []
    for c in class_dirs:
        p = os.path.join(BASE_DIR, c)
        for f in os.listdir(p):
            if f.endswith(".txt"):
                filepath = os.path.join(p, f)
                # lecture texte
                try:
                    with open(filepath, "r", encoding="utf-8") as txtf:
                        content = txtf.read()
                except:
                    content = "(Erreur lecture texte)"

                # infos fichier
                size_kb = os.path.getsize(filepath) / 1024
                date_modif = datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%d/%m/%Y %H:%M:%S")

                all_docs.append({
                    "classe": c,
                    "nom": f,
                    "taille": f"{size_kb:.1f} Ko",
                    "date": date_modif,
                    "texte": content
                })

    if not all_docs:
        st.info("Aucun document sauvegardé pour le moment.")
    else:
        import pandas as pd
        df = pd.DataFrame(all_docs)
        st.dataframe(df[["classe","nom","taille","date"]], use_container_width=True)

        choix = st.selectbox("📑 Choisir un document pour le détail :", [d["nom"] for d in all_docs])

        if choix:
            doc = next(d for d in all_docs if d["nom"] == choix)

            st.subheader(f"Détails — {doc['nom']}")
            st.markdown(f"**Classe :** {doc['classe']}")
            st.markdown(f"**Taille :** {doc['taille']}")
            st.markdown(f"**Date :** {doc['date']}")

            with st.expander("📝 Texte extrait"):
                st.text_area("Contenu OCR", doc["texte"], height=300)

            # Image associée
            img_name = doc["nom"].replace(".txt", ".png")
            img_path = os.path.join(BASE_DIR, doc["classe"], img_name)
            if os.path.exists(img_path):
                st.image(img_path, caption="Image associée", use_container_width=True)
            else:
                st.warning("⚠️ Aucune image associée trouvée pour ce document.")

# -------------------------
# ASSISTANT IA (chat)
# -------------------------
with tab_assistant:
    
    st.header("DocuVision IA — Documentation administrative (Cameroun)")
    st.caption("Réponses restreintes au domaine (arrêtés, décrets, circulaires, en-têtes, cachets, référencements…).")

    api_key = os.getenv("HF_TOKEN")

    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header"><span class="chat-title">DocuVision IA</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-body">', unsafe_allow_html=True)

    # Initialisation de l'historique des conversations
    if "history" not in st.session_state:
        st.session_state.history = []

    # Zone de saisie pour l'utilisateur
    user_question = st.text_input("Posez votre question sur les documents administratifs :")

    # Clé API Hugging Face
    API_KEY = os.getenv("HF_TOKEN")
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    HEADERS = {"Authorization": f"Bearer {API_KEY}"}

    def ask_admin_documents_question(question, history):
        """
        Envoie la question à l'API Hugging Face en demandant au modèle de 
        répondre uniquement sur les documents administratifs.
        """
        system_prompt = (
            "Tu es un assistant strictement spécialisé dans les documents administratifs, les documents, l'administration, les circulaires, les decrets. "
            "Ne réponds qu'aux questions liées aux documents administratifs et soit toujours poli. "
            "Si la question n'a rien à voir avec ce domaine, réponds : "
            "'Je ne peux répondre qu’aux questions sur les documents administratifs.'"
        )
        
        # Messages à envoyer à l'API
        messages = [{"role": "system", "content": system_prompt}]
        
        # Ajouter uniquement le dernier échange (user + assistant) pour le contexte
        if history:
            last_entry = history[-1]
            messages.append({"role": "user", "content": last_entry["user"]})
            messages.append({"role": "assistant", "content": last_entry["assistant"]})
        
        # Ajouter la nouvelle question
        messages.append({"role": "user", "content": question})
        
        payload = {"messages": messages, "model": "google/gemma-2-2b-it:nebius"}
        
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Erreur {response.status_code}: {response.text}"

    # Bouton pour envoyer la question
    if st.button("Envoyer") and user_question:
        with st.spinner("Le modèle réfléchit..."):
            answer = ask_admin_documents_question(user_question, st.session_state.history)
            
            # Sauvegarder la conversation dans l'historique
            st.session_state.history.append({
                "user": user_question,
                "assistant": answer
            })

    # Affichage de l'historique inversé (les derniers messages en haut)
    if st.session_state.history:
        st.subheader("🕒 Historique des conversations")
        for entry in reversed(st.session_state.history):
            st.markdown(f" 👤 : {entry['user']}")
            st.markdown(f" 🤖 : {entry['assistant']}")
            st.markdown("---")


# =========================
# FIN
# =========================
