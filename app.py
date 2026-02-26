import streamlit as st
import os
import pandas as pd
import json
import tempfile
import hashlib
from dotenv import load_dotenv
from PIL import Image
import extractor
from streamlit_extras.colored_header import colored_header

# Try to import hybrid extractor and learning modules
try:
    import hybrid_extractor
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

try:
    from invoice_db import get_database
    from confidence_scorer import calculate_confidence, get_confidence_summary, needs_human_review
    from template_generator import generate_template, should_generate_template
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="AI Invoice Parser",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown(
    """
<style>
    .main { background-color: #f5f7fa; }
    .stButton>button {
        width: 100%;
        border-radius: 999px;
        height: 3em;
        background: linear-gradient(90deg, #22c55e, #16a34a);
        color: white;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #16a34a, #15803d);
        box-shadow: 0 10px 25px rgba(22, 163, 74, 0.35);
    }
    h1 { color: #0f172a; }
    h2, h3 { color: #1f2937; }
    .confidence-high { color: #22c55e; font-weight: bold; }
    .confidence-medium { color: #f59e0b; font-weight: bold; }
    .confidence-low { color: #ef4444; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)

# Application Header
colored_header(
    label="AI Invoice Parser",
    description="Upload an invoice and let AI extract data. The system learns and improves from your corrections.",
    color_name="green-70",
)

# Provider selection
col_provider, col_status, col_stats = st.columns([1, 2, 1])
with col_provider:
    provider = st.selectbox("AI Provider", ["Gemini", "OpenAI", "Claude"], index=0, label_visibility="collapsed")

# Get API key
if provider == "Gemini":
    api_key_env = os.getenv("GEMINI_API_KEY")
    model_name = "gemini-2.5-flash"
elif provider == "OpenAI":
    api_key_env = os.getenv("OPENAI_API_KEY")
    model_name = "gpt-4o-mini"
else: # Claude
    api_key_env = os.getenv("CLAUDE_API_KEY")
    # model_name = "claude-3-5-sonnet-20240620"
    model_name = "claude-3-haiku-20240307"

api_key = api_key_env
with col_status:
    if not api_key:
        api_key = st.text_input(f"Enter {provider} API Key", type="password", label_visibility="collapsed", placeholder=f"Enter {provider} API Key")
        if not api_key:
            if provider == "Gemini":
                st.caption("⚠️ [Get Gemini Key](https://aistudio.google.com/app/apikey)")
            elif provider == "OpenAI":
                st.caption("⚠️ [Get OpenAI Key](https://platform.openai.com/api-keys)")
            else:
                st.caption("⚠️ [Get Claude Key](https://console.anthropic.com/)")
    else:
        st.caption(f"✅ {provider} ready")

# Show database stats
with col_stats:
    if LEARNING_AVAILABLE:
        db = get_database()
        stats = db.get_stats()
        st.caption(f"📊 {stats['total_extractions']} extractions | {stats['total_templates']} templates")

use_hybrid = HYBRID_AVAILABLE

st.markdown("---")

# Initialize session state
if "extraction_data" not in st.session_state:
    st.session_state.extraction_data = None
if "extraction_id" not in st.session_state:
    st.session_state.extraction_id = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None
if "current_page_index" not in st.session_state:
    st.session_state.current_page_index = 0

# File Uploader
st.markdown("### 1. Upload document")
document_type = st.selectbox(
    "Document Type",
    ["Purchase Order bill", "Sales Invoice bill", "Sales Order bill"],
    help="Select the type of document you are uploading."
)

uploaded_file = st.file_uploader(
    "Drop your document here or click to browse",
    type=["pdf", "jpg", "jpeg", "png"],
    help="Supports PDF and image formats.",
)

if uploaded_file is not None and api_key:
    # Clear extraction data if a new file is uploaded
    if st.session_state.current_file_id != uploaded_file.file_id:
        st.session_state.current_file_id = uploaded_file.file_id
        st.session_state.extraction_data = None
        st.session_state.extraction_id = None
        st.session_state.confidence = None
        st.session_state.edit_mode = False
        st.session_state.current_page_index = 0

    st.markdown("### 2. Review & extract")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### Preview")
        image_to_process = None

        if uploaded_file.type == "application/pdf":
            with st.spinner("Rendering PDF..."):
                images = extractor.convert_pdf_to_images(uploaded_file.read())
                if images:
                    image_to_process = images
                    total_pages = len(images)
                    # Ensure current page index is in range
                    if st.session_state.current_page_index >= total_pages:
                        st.session_state.current_page_index = 0

                    current_idx = st.session_state.current_page_index
                    current_img = images[current_idx]

                    st.image(current_img, use_container_width=True, caption=f"Page {current_idx + 1} of {total_pages}")

                    prev_col, info_col, next_col = st.columns([1, 2, 1])
                    with prev_col:
                        if st.button("⬅ Back", disabled=current_idx == 0, key="prev_page_btn"):
                            st.session_state.current_page_index = max(0, current_idx - 1)
                            st.rerun()
                    with info_col:
                        st.markdown(f"<div style='text-align:center;'>Page {current_idx + 1} of {total_pages}</div>", unsafe_allow_html=True)
                    with next_col:
                        if st.button("Next ➡", disabled=current_idx == total_pages - 1, key="next_page_btn"):
                            st.session_state.current_page_index = min(total_pages - 1, current_idx + 1)
                            st.rerun()
                else:
                    st.error("Failed to convert PDF.")
        else:
            image_to_process = Image.open(uploaded_file)
            st.image(image_to_process, use_container_width=True)

    with col2:
        st.markdown("#### Extraction results")
        if image_to_process:
            extract_btn = st.button("✨ Extract data", type="primary")

            if extract_btn:
                # Reset edit mode on new extraction
                st.session_state.edit_mode = False
                
                if use_hybrid and HYBRID_AVAILABLE:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        with st.spinner(f"Processing with {provider}..."):
                            data = hybrid_extractor.hybrid_extract_invoice(
                                tmp_path, image_to_process, api_key,
                                provider=provider, model_name=model_name,
                                document_type=document_type,
                            )
                            st.session_state.extraction_data = data
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                else:
                    with st.spinner(f"Processing with {provider}..."):
                        data = extractor.extract_invoice_data(
                            image_to_process, api_key,
                            provider=provider, model_name=model_name,
                            document_type=document_type,
                        )
                        st.session_state.extraction_data = data
                
                # Calculate confidence and save to database
                if LEARNING_AVAILABLE and st.session_state.extraction_data:
                    display_data = st.session_state.extraction_data.get("final_data", st.session_state.extraction_data)
                    if "error" not in display_data:
                        confidence_result = calculate_confidence(display_data)
                        st.session_state.confidence = confidence_result
                        
                        # Generate image hash for deduplication
                        if isinstance(image_to_process, list):
                            img_bytes = image_to_process[0].tobytes()
                        else:
                            img_bytes = image_to_process.tobytes()
                        image_hash = hashlib.md5(img_bytes[:10000]).hexdigest()
                        
                        # Save extraction
                        db = get_database()
                        extraction_id = db.save_extraction(
                            display_data,
                            confidence=confidence_result["overall"],
                            source="hybrid" if use_hybrid else "llm",
                            image_hash=image_hash,
                        )
                        st.session_state.extraction_id = extraction_id
                        
                        # Auto-generate template if high confidence
                        if should_generate_template(display_data, confidence_result["overall"]):
                            vendor_name = display_data.get("vendor_name")
                            if vendor_name and not db.has_template(vendor_name):
                                template = generate_template(vendor_name, display_data)
                                db.save_template(vendor_name, template)

            # Display results
            data = st.session_state.extraction_data
            if data:
                if "error" in data:
                    st.error(f"Extraction failed: {data['error']}")
                else:
                    display_data = data.get("final_data", data) if (use_hybrid and "final_data" in data) else data

                    # Show confidence score
                    if LEARNING_AVAILABLE and st.session_state.confidence:
                        conf = st.session_state.confidence
                        overall = conf["overall"]
                        
                        if overall >= 0.9:
                            st.success(f"✅ High Confidence ({overall:.0%})")
                        elif overall >= 0.75:
                            st.warning(f"⚠️ Medium Confidence ({overall:.0%}) - Review recommended")
                        else:
                            st.error(f"❌ Low Confidence ({overall:.0%}) - Corrections needed")
                        
                        # Show flags
                        if conf["flags"]:
                            with st.expander(f"⚠️ {len(conf['flags'])} issue(s) detected"):
                                for flag in conf["flags"]:
                                    st.warning(flag)
                    
                    # Edit mode toggle (checkbox with key so Streamlit preserves state on click)
                    if "edit_mode" not in st.session_state:
                        st.session_state.edit_mode = False
                    st.session_state.edit_mode = st.checkbox(
                        "✏️ Edit & Correct",
                        value=st.session_state.edit_mode,
                        key="edit_mode_checkbox",
                    )
                    
                    if st.session_state.edit_mode:
                        # Editable form
                        st.markdown("##### Edit extracted data")
                        st.caption("Change any field below and click **Save Corrections**. (Maps to common fields)")
                        
                        # Use first available field from various schemas mapping to common concepts
                        doc_number = display_data.get("invoice_number") or display_data.get("po_number") or display_data.get("order_number") or ""
                        doc_date = display_data.get("date") or display_data.get("date_of_issue") or display_data.get("order_date") or ""
                        
                        summary_data = display_data.get("summary", {})
                        doc_total = (
                            display_data.get("total_amount") or 
                            display_data.get("total_amount_due") or 
                            display_data.get("order_total") or 
                            summary_data.get("total_amount_due") or
                            summary_data.get("order_total") or
                            summary_data.get("total_commitment") or 0
                        )
                        doc_vendor = display_data.get("vendor_name") or display_data.get("from") or display_data.get("seller") or ""
                        
                        with st.form("correction_form", clear_on_submit=False):
                            edited_invoice_number = st.text_input("Document Number", value=str(doc_number))
                            edited_date = st.text_input("Date (YYYY-MM-DD)", value=str(doc_date))
                            edited_currency = st.text_input("Currency", value=str(display_data.get("currency") or ""))
                            edited_amount = st.number_input("Total Amount", value=float(doc_total))
                            edited_vendor = st.text_input("Provider/Vendor", value=str(doc_vendor))
                            
                            # Summary fields
                            st.markdown("##### Summary")
                            summary = display_data.get("summary", {})
                            sub = summary.get("subtotal") or summary.get("order_subtotal") or summary.get("total_usage_sale_charges") or 0

                            # Derive a tax AMOUNT, not percentage
                            tax = None

                            # 1) Prefer explicit tax fields from summary
                            if summary.get("tax") is not None:
                                tax = summary.get("tax")
                            elif summary.get("tax_amount") is not None:
                                tax = summary.get("tax_amount")

                            # 2) If hybrid extractor provided additional_info.tax_amount, prefer that next
                            if tax is None:
                                add_info = display_data.get("additional_info", {})
                                if isinstance(add_info, dict) and add_info.get("tax_amount") is not None:
                                    tax = add_info.get("tax_amount")

                            # 3) For Sales Invoice, convert percentage to amount if needed
                            if tax is None and document_type == "Sales Invoice bill":
                                tax_pct = summary.get("tax_percentage")
                                if isinstance(tax_pct, (int, float)) and isinstance(sub, (int, float)):
                                    tax = float(sub) * float(tax_pct) / 100.0

                            # 4) Fallback: sum line-item taxes
                            if tax is None:
                                lines = display_data.get("line_items") or display_data.get("charges") or display_data.get("order_items") or []
                                tax = sum((item.get("tax_amount") or 0) for item in lines if isinstance(item, dict))

                            edited_subtotal = st.number_input("Subtotal/Charges", value=float(sub))
                            tax_help = "Derived from summary / additional_info when available, otherwise sum of line-item taxes."
                            edited_tax = st.number_input("Tax", value=float(tax or 0), help=tax_help)
                            
                            submitted = st.form_submit_button("💾 Save Corrections")
                            
                            if submitted:
                                # Create corrected data
                                corrected_data = display_data.copy()
                                corrected_data["invoice_number"] = edited_invoice_number
                                corrected_data["date"] = edited_date
                                corrected_data["currency"] = edited_currency
                                corrected_data["total_amount"] = edited_amount
                                corrected_data["vendor_name"] = edited_vendor
                                corrected_data["summary"] = corrected_data.get("summary", {})
                                corrected_data["summary"]["subtotal"] = edited_subtotal
                                corrected_data["summary"]["tax"] = edited_tax
                                
                                # Find changed fields (simplify for common fields)
                                changed_fields = []
                                if edited_invoice_number != str(doc_number):
                                    changed_fields.append("invoice_number")
                                if edited_date != str(doc_date):
                                    changed_fields.append("date")
                                if edited_currency != display_data.get("currency"):
                                    changed_fields.append("currency")
                                if edited_amount != float(doc_total):
                                    changed_fields.append("total_amount")
                                if edited_vendor != str(doc_vendor):
                                    changed_fields.append("vendor_name")
                                
                                # Save correction to database
                                if LEARNING_AVAILABLE and st.session_state.extraction_id and changed_fields:
                                    db = get_database()
                                    db.save_correction(
                                        st.session_state.extraction_id,
                                        display_data,
                                        corrected_data,
                                        changed_fields,
                                    )
                                    st.success(f"✅ Saved corrections for: {', '.join(changed_fields)}")
                                    
                                    # Update session data
                                    if "final_data" in st.session_state.extraction_data:
                                        st.session_state.extraction_data["final_data"] = corrected_data
                                    else:
                                        st.session_state.extraction_data = corrected_data
                                    
                                    st.session_state.edit_mode = False
                                    if "edit_mode_checkbox" in st.session_state:
                                        st.session_state.edit_mode_checkbox = False
                                    st.rerun()
                                elif not changed_fields:
                                    st.info("No changes detected")
                    else:
                        # View mode - show metrics
                        m1, m2, m3 = st.columns(3)
                        
                        doc_number = display_data.get("invoice_number") or display_data.get("po_number") or display_data.get("order_number") or ""
                        doc_date = display_data.get("date") or display_data.get("date_of_issue") or display_data.get("order_date") or ""
                        
                        summary_data = display_data.get("summary", {})
                        doc_total = (
                            display_data.get("total_amount") or 
                            display_data.get("total_amount_due") or 
                            display_data.get("order_total") or 
                            summary_data.get("total_amount_due") or
                            summary_data.get("order_total") or
                            summary_data.get("total_commitment") or 0
                        )
                        doc_vendor = display_data.get("vendor_name") or display_data.get("from") or display_data.get("seller") or ""
                        
                        m1.metric("Total", f"{doc_total} {display_data.get('currency', '')}")
                        m2.metric("Date", doc_date or "N/A")
                        m3.metric("Document #", doc_number or "N/A")
                        
                        st.success(f"Provider/Vendor: **{doc_vendor or 'Unknown'}**")
                        
                        # Summary
                        summary = display_data.get("summary", {})
                        if summary and any(v is not None for v in summary.values()):
                            with st.expander("📊 Invoice Summary", expanded=True):
                                s1, s2 = st.columns(2)
                                if summary.get("subtotal") is not None:
                                    s1.metric("Subtotal", f"{summary.get('subtotal')} {display_data.get('currency', '')}")

                                # Derive a tax AMOUNT, not percentage
                                tax_val = None

                                # 1) Prefer explicit tax fields from summary
                                if summary.get("tax") is not None:
                                    tax_val = summary.get("tax")
                                elif summary.get("tax_amount") is not None:
                                    tax_val = summary.get("tax_amount")

                                # 2) If hybrid extractor provided additional_info.tax_amount, prefer that next
                                if tax_val is None:
                                    add_info = display_data.get("additional_info", {})
                                    if isinstance(add_info, dict) and add_info.get("tax_amount") is not None:
                                        tax_val = add_info.get("tax_amount")

                                # 3) For Sales Invoice, convert percentage to amount if needed
                                if tax_val is None and document_type == "Sales Invoice bill":
                                    subtotal_for_tax = (
                                        summary.get("subtotal")
                                        or summary.get("order_subtotal")
                                        or summary.get("total_usage_sale_charges")
                                    )
                                    tax_pct = summary.get("tax_percentage")
                                    if isinstance(tax_pct, (int, float)) and isinstance(subtotal_for_tax, (int, float)):
                                        tax_val = float(subtotal_for_tax) * float(tax_pct) / 100.0

                                # 4) Fallback: sum line-item taxes
                                if tax_val is None:
                                    lines = display_data.get("line_items") or display_data.get("charges") or display_data.get("order_items") or []
                                    tax_val = sum((item.get("tax_amount") or 0) for item in lines if isinstance(item, dict))

                                if tax_val is not None:
                                    tax_help = "Derived from summary / additional_info when available, otherwise sum of line-item taxes."
                                    s2.metric("Tax", f"{tax_val} {display_data.get('currency', '')}", help=tax_help)
                                
                                if summary.get("billing_period"):
                                    st.text(f"Billing Period: {summary.get('billing_period')}")
                                if summary.get("account_number"):
                                    st.text(f"Account #: {summary.get('account_number')}")
                        
                        # Tabs
                        tab1, tab2, tab3 = st.tabs(["📝 Line Items", "📊 Details", "🧠 Learning"])
                        
                        with tab1:
                            lines = display_data.get("line_items") or display_data.get("charges") or display_data.get("order_items")
                            if lines and isinstance(lines, list):
                                df = pd.DataFrame(lines)
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.info("No line items found.")
                        
                        with tab2:
                            st.json(display_data)
                        
                        with tab3:
                            if LEARNING_AVAILABLE:
                                db = get_database()
                                
                                # Show extraction info
                                if st.session_state.extraction_id:
                                    st.caption(f"Extraction ID: {st.session_state.extraction_id}")
                                
                                # Show if template exists for this vendor
                                vendor = display_data.get("vendor_name")
                                if vendor:
                                    if db.has_template(vendor):
                                        st.success(f"✅ Template exists for {vendor}")
                                        if st.button("View Template"):
                                            template = db.get_template(vendor)
                                            st.code(template, language="yaml")
                                    else:
                                        st.info(f"No template yet for {vendor}")
                                        if st.button("🔧 Generate Template"):
                                            template = generate_template(vendor, display_data)
                                            db.save_template(vendor, template)
                                            st.success("Template generated!")
                                            st.code(template, language="yaml")
                                
                                # Show past extractions for this vendor
                                if vendor:
                                    past = db.get_vendor_extractions(vendor, limit=5)
                                    if len(past) > 1:
                                        st.markdown(f"##### Past extractions from {vendor}")
                                        for ext in past[-3:]:
                                            st.caption(f"• {ext['timestamp'][:10]} - ${ext['data'].get('total_amount', 'N/A')}")
                            else:
                                st.info("Learning features not available")
                        
                        # Download
                        json_str = json.dumps(display_data, indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name="invoice_data.json",
                            mime="application/json"
                        )
        else:
            st.info("Upload an invoice to enable extraction.")
