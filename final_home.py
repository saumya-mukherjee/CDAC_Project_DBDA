import cv2
import numpy as np
import streamlit as st
# We import from our new, combined analysis file
from final_analyze import (
    # --- Your 5 Original Functions ---
    sentence_alignment,
    detect_tall_narrow_letters,
    detect_writing_pressure,
    detect_alignment,
    detect_handwriting_size,
    
    # --- 5 New Enhanced Functions (Now Fixed) ---
    detect_letter_slant,
    detect_word_spacing,
    detect_line_spacing,
    detect_t_bars,
    detect_i_dots
)

# Set Streamlit page configuration
st.set_page_config(page_title="Enhanced English Handwriting Analysis", layout="wide")

st.title("✒️ Enhanced English Handwriting Analysis Tool (V-Final)")
st.write("This version uses your 5 original features (with bugs fixed) and adds 5 new advanced features.")

# File uploader
uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_color = cv2.imdecode(file_bytes, 1)
    
    # Also create a grayscale version, as many functions need it
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    st.image(img_color, channels="BGR", caption="📥 Your Uploaded Image", use_column_width=True)
    
    st.sidebar.title("Analysis Options")
    # Analysis options in English
    option = st.sidebar.radio(
        "Which feature would you like to analyze?",
        [
            # --- Your 5 Original Features ---
            "Sentence Alignment",
            "Tall & Narrow Letters",
            "Writing Pressure",
            "Text Alignment",
            "Handwriting Size",
            
            # --- 5 New Enhanced Features ---
            "Letter Slant (New)",
            "Word Spacing (New)",
            "Line Spacing (New)",
            "T-Bar Analysis (New)",
            "I-Dot Analysis (New)"
        ]
    )

    st.markdown("---")
    st.header(f"Analysis Results: {option}")

    # --- 1. Sentence Alignment (Your Original Function) ---
    if option.startswith("Sentence Alignment"):
        processed_img, results, threshold = sentence_alignment(img_color.copy())
        st.image(processed_img, channels="BGR", caption="Sentence Alignment Analysis (Your Original Code)")
        st.write("### Results")
        st.write(f"Adaptive straight threshold: ±{threshold:.4f}")
        st.write(f"Total {len(results)} lines detected.")
        
        results_data = []
        for res in results:
            results_data.append({
                "Line Number": res['line_number'],
                "Slope": f"{res['slope']:.4f}",
                "Direction": res['direction'],
                "Word Count": res['word_count']
            })
        st.table(results_data)

        st.write("### Behavioral Analysis")
        for res in results:
            if res['direction'] == "upward":
                st.write(f"**Line {res['line_number']} (Upward):** Suggests optimism, motivation, and confidence.")
            elif res['direction'] == "downward":
                st.write(f"**Line {res['line_number']} (Downward):** Reflects fatigue, discouragement, or low energy.")
            else:
                st.write(f"**Line {res['line_number']} (Straight):** Shows emotional balance, calmness, and logical thinking.")

    # --- 2. Tall & Narrow Letters (Your Original Function) ---
    elif option.startswith("Tall & Narrow Letters"):
        processed_img, count = detect_tall_narrow_letters(img_color.copy())
        st.image(processed_img, channels="BGR", caption="Tall & Narrow Letters Result (Your Original Code)")
        st.write("### RESULT")
        st.write(f"Total tall & narrow letters detected: {count}")

        st.write("### Behaviour Analysis")
        if count > 0:
            st.write("The presence of tall and narrow letters suggests **ambition, focus, and eagerness for recognition**.")
        else:
            st.write("No tall & narrow letters detected. This may indicate a more **modest, balanced writing style**.")

    # --- 3. Writing Pressure (Your Original Function) ---
    elif option.startswith("Writing Pressure"):
        pressure, thickness, intensity = detect_writing_pressure(img_color.copy())
        st.write("### Results")
        st.write(f"**Pressure Type:** {pressure}")
        st.write(f"**Average Stroke Thickness:** {thickness:.2f} (higher = heavier)")
        st.write(f"**Average Stroke Darkness:** {intensity:.2f} (lower = heavier)")

        st.write("### Behavioral Analysis")
        if pressure == "Heavy":
            st.write("✍️ **Heavy Pressure:** Suggests strong determination, powerful emotions, and persistence. Can be a sign of high energy.")
        else:
            st.write("🖋️ **Light Pressure:** Suggests sensitivity, relaxation, or low stress. It can also be a sign of flexibility.")

    # --- 4. Text Alignment (Your Original Function - FIXED) ---
    elif option.startswith("Text Alignment"):
        # --- FIX: Your original function only returns ONE string. ---
        # Your original home.py expected 3 values, which caused a crash.
        # This is now fixed to handle the 1 string your analyze.py returns.
        alignment_result = detect_alignment(img_gray.copy()) 
        
        st.image(img_color, channels="BGR", caption="Text Alignment (Your Original Code)")
        
        st.write("### Results")
        st.write(f"**Margin Alignment:** {alignment_result}")

        st.write("### Behavioral Analysis")
        if alignment_result == "Left Aligned":
            st.write("Left-aligned text suggests **organization and adherence to rules**.")
        elif alignment_result == "Right Aligned":
            st.write("Right-aligned text suggests **unconventional thinking or a rebellious nature**.")
        elif alignment_result == "Center Aligned":
            st.write("Center-aligned text suggests a **desire for attention or a need for balance**.")
        else:
            st.write(f"{alignment_result}") # Catches "No text detected"

    # --- 5. Handwriting Size (Your Original Function - FIXED) ---
    elif option.startswith("Handwriting Size"):
        # --- FIX: Your original function only returns ONE string. ---
        # Your original home.py expected 3 values, which caused a crash.
        # This is now fixed to handle the 1 string your analyze.py returns.
        size_result = detect_handwriting_size(img_gray.copy())
        
        st.image(img_color, channels="BGR", caption="Handwriting Size (Your Original Code)")
        
        st.write("### Results")
        st.write(f"**Overall Size:** {size_result}")

        st.write("### Behavioral Analysis")
        if size_result == "Large Handwriting":
            st.write("Large handwriting suggests an **outgoing, confident, and people-oriented personality**.")
        elif size_result == "Small Handwriting":
            st.write("Small handwriting suggests **introversion, strong focus, and attention to detail**.")
        # Your original code did not return "Medium", so I am matching its logic.
        elif size_result == "No handwriting detected":
             st.write("Could not detect handwriting to analyze size.")
        else:
            st.write(f"Analysis result: {size_result}")

    # --- 6. Letter Slant (New) ---
    elif option.startswith("Letter Slant"):
        processed_img, slant_category, avg_angle = detect_letter_slant(img_color.copy())
        st.image(processed_img, channels="BGR", caption="Letter Slant Analysis (New)")
        st.write("### Results")
        st.write(f"**Slant Type:** {slant_category}")
        st.write(f"**Average Angle:** {avg_angle:.2f} degrees")

        st.write("### Behavioral Analysis")
        if slant_category == "Right Slant":
            st.write("A right slant suggests an emotionally expressive, friendly, and future-oriented personality.")
        elif slant_category == "Left Slant":
            st.write("A left slant may indicate an introverted, reserved personality that is focused on the past.")
        elif slant_category == "Vertical":
            st.write("Vertical writing suggests emotional control, independence, and logical thinking.")
        else:
            st.write("A mixed slant can indicate an unstable emotional state or internal conflict.")

    # --- 7. Word Spacing (New) ---
    elif option.startswith("Word Spacing"):
        processed_img, spacing_category, avg_distance = detect_word_spacing(img_color.copy())
        st.image(processed_img, channels="BGR", caption="Word Spacing Analysis (New)")
        st.write("### Results")
        st.write(f"**Spacing Type:** {spacing_category}")
        st.write(f"**Average Distance Between Words:** {avg_distance:.2f} pixels")

        st.write("### Behavioral Analysis")
        if spacing_category == "Wide":
            st.write("Wide spacing between words can indicate mental clarity, a need for freedom, or loneliness.")
        elif spacing_category == "Narrow":
            st.write("Narrow spacing between words can indicate haste, restlessness, or a desire to be with others.")
        else:
            st.write("Consistent spacing indicates a balanced and logical mind.")

    # --- 8. Line Spacing (New - FIXED) ---
    elif option.startswith("Line Spacing"):
        # This function is now fixed and will not crash
        processed_img, spacing_category, avg_distance = detect_line_spacing(img_color.copy())
        st.image(processed_img, channels="BGR", caption="Line Spacing Analysis (New)")
        st.write("### Results")
        st.write(f"**Line Spacing:** {spacing_category}")
        st.write(f"**Average Distance Between Lines:** {avg_distance:.2f} pixels")

        st.write("### Behavioral Analysis")
        if spacing_category == "Wide":
            st.write("Wide spacing between lines indicates clarity of thought, organization, and a capacity to see things separately.")
        elif spacing_category == "Narrow":
            st.write("Narrow spacing (or tangled lines) can indicate confused thoughts, stress, or a tendency to rush.")
        else:
            st.write("Consistent spacing indicates a balanced, logical, and disciplined mind.")

    # --- 9. T-Bar Analysis (New) ---
    elif option.startswith("T-Bar Analysis"):
        processed_img, report = detect_t_bars(img_color.copy())
        st.image(processed_img, channels="BGR", caption="T-Bar Analysis (New)")
        st.write("### Results")
        st.write(f"**T-Bars Detected:** {report['t_bars_found']}")
        st.write(f"**Dominant Placement:** {report['dominant_placement']}")
        
        st.write("### Behavioral Analysis")
        if report['dominant_placement'] == "High on Stem":
            st.write("High t-bars (crossed near the top) suggest ambition, high self-esteem, and optimism.")
        elif report['dominant_placement'] == "Low on Stem":
            st.write("Low t-bars (crossed near the bottom) suggest insecurity, low self-esteem, or a lack of ambition.")
        elif report['dominant_placement'] == "Middle of Stem":
            st.write("Middle t-bars (crossed in the center) suggest a balanced, practical, and well-adjusted personality.")
        elif report['dominant_placement'] == "Variable":
            st.write("Variable placement suggests a lack of decisiveness or fluctuating moods.")
        else:
            st.write("Could not determine a dominant t-bar placement.")

    # --- 10. I-Dot Analysis (New) ---
    elif option.startswith("I-Dot Analysis"):
        processed_img, report = detect_i_dots(img_color.copy())
        st.image(processed_img, channels="BGR", caption="I-Dot Analysis (New)")
        st.write("### Results")
        st.write(f"**I-Dots Detected:** {report['i_dots_found']}")
        st.write(f"**Dominant Placement:** {report['dominant_placement']}")
        st.write(f"**Dominant Shape:** {report['dominant_shape']}")

        st.write("### Behavioral Analysis")
        if report['dominant_placement'] == "Precisely Above":
            st.write("**Placement:** A precisely placed dot suggests a detail-oriented, meticulous, and focused mind.")
        elif report['dominant_placement'] == "Far Above":
            st.write("**Placement:** A high, floating dot suggests a strong imagination, creativity, and an idealistic nature.")
        elif report['dominant_placement'] == "To the Right":
            st.write("**Placement:** A dot to the right suggests impatience and forward-thinking.")
        elif report['dominant_placement'] == "To the Left":
            st.write("**Placement:** A dot to the left suggests caution, deliberation, or looking to the past.")
        
        if report['dominant_shape'] == "Dot":
            st.write("**Shape:** A simple, firm dot reinforces precision and focus.")
        elif report['dominant_shape'] == "Circle":
            st.write("**Shape:** A circular dot suggests creativity, a desire to be unique, or sometimes immaturity.")
        elif report['dominant_shape'] == "Slash":
            st.write("**Shape:** A slash or dash for a dot suggests high energy, impatience, or a critical nature.")

else:
    st.info("Please upload an image to begin.")

st.sidebar.markdown("---")
st.sidebar.info("© 2025 Enhanced Handwriting Analyzer (V-Final)")