##---------------------------------------------Imports---------------------------------------------##

import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
from fpdf import FPDF
import tempfile

##---------------------------------------------Chart COnfiguration---------------------------------------------##

import matplotlib.pyplot as plt
import io

def plot_quadrant_chart(axis_labels, quadrants):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')

    # Axis labels
    ax.text(0.5, 1.05, axis_labels["Y-Axis"], ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, -0.1, axis_labels["-Y-Axis"], ha='center', va='top', fontsize=12, transform=ax.transAxes)
    ax.text(1.02, 0.5, axis_labels["X-Axis"], ha='left', va='center', fontsize=12, rotation=270, transform=ax.transAxes)
    ax.text(-0.05, 0.5, axis_labels["-X-Axis"], ha='right', va='center', fontsize=12, rotation=90, transform=ax.transAxes)

    # Quadrants
    positions = {
        "Q1": (0.5, 0.5),
        "Q2": (-0.5, 0.5),
        "Q3": (-0.5, -0.5),
        "Q4": (0.5, -0.5)
    }

    for q, (x, y) in positions.items():
        companies = "\n".join(quadrants.get(q, []))
        ax.text(x, y, companies, ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightgrey", ec="black"))

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    return fig

import unicodedata

def sanitize_text(text):
    return unicodedata.normalize('NFKD', text).encode('latin-1', 'ignore').decode('latin-1')

##---------------------------------------------PDF---------------------------------------------##



##---------------------------------------------Start AI MAtrix Creation---------------------------------------------##


# ✅ Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ Industry filter
SELECTED_INDUSTRIES = [
    "Fintech", "Artificial intelligence & machine learning (AI&ML)", "Lifestyles of Health Sustainability (LOHAS)",
    "Digital health", "Software as a service (SaaS)", "Gaming", "Cryptocurrency", "HR tech", "Cloudtech",
    "Supply chain", "Future of Work", "Marketing tech", "Edtech", "Foodtech", "Real estate tech",
    "Ecommerce", "Cleantech", "Healthtech", "Insurtech"
]

@st.cache_data
def load_data():
    df = pd.read_excel("Interface_Data.xlsx", sheet_name="Step 1 All Portcos")

    industry_cols = [
        "Primary Industry Code", "Industry Code 2", "Industry Code 3",
        "Industry Code 4", "Industry Code 5", "Industry Code 6", "Industry Code 7"
    ]

    df['Matched Industries'] = df[industry_cols].apply(
        lambda row: [industry for industry in row if isinstance(industry, str) and industry.strip() in SELECTED_INDUSTRIES],
        axis=1
    )

    df = df[df['Matched Industries'].map(len) > 0].copy()
    return df

@st.cache_data
def load_example_prompts():
    with open("example_one.txt", "r", encoding="utf-8") as f1, open("example_two.txt", "r", encoding="utf-8") as f2:
        return f1.read(), f2.read()

# ==========================
# ✅ Streamlit App Interface
# ==========================
st.title("Industry Matrix Generator")

# Load data and examples
df = load_data()
example_one, example_two = load_example_prompts()

# User selects an industry
selected_industry = st.selectbox("Choose an industry to analyze:", SELECTED_INDUSTRIES)

##---------------------------------------------Graph Creation---------------------------------------------##
def extract_labels_and_quadrants(gpt_text):
    import re

    # Step 1: Match all 4 intersection headers
    headers = re.findall(r"\*\*(.+?) <> (.+?):", gpt_text)
    if len(headers) < 4:
        return {}, {}

    # Define axis labels per your preferred order
    axis_labels = {
        "X-Axis": headers[3][1].strip(),   # Q4
        "-X-Axis": headers[1][1].strip(),  # Q2
        "Y-Axis": headers[0][1].strip(),   # Q1
        "-Y-Axis": headers[2][1].strip(),  # Q3
    }

    # Step 2: Split full text into 4 quadrant blocks (after each intersection header)
    intersection_blocks = re.split(r"\*\*(.+?) <> (.+?):.*\*\*", gpt_text)[1:]

    quadrants = {}

    for i in range(0, len(intersection_blocks), 3):
        if i + 2 >= len(intersection_blocks):
            continue

        q_label = f"Q{(i // 3) + 1}"
        paragraph = intersection_blocks[i + 2]

        # 1. Try Markdown hyperlink
        match = re.search(r"\[([^\]]+)\]\((https?://[^\)]+)\)", paragraph)
        if match:
            company_name = match.group(1).strip()
        else:
            # 2. Try "XYZ is a [example|leader|etc]"
            fallback = re.search(
                r"\b([A-Z][A-Za-z0-9 &|]{2,80})\b\s+(?:is|are|has|have|offers|provides)",
                paragraph
            )
            if fallback:
                company_name = fallback.group(1).strip()
            else:
                # 3. Final fallback: first capitalized phrase in first bullet
                bullets = re.findall(r"[-–•]\s+(.*)", paragraph)
                if bullets:
                    cap_phrase = re.search(r"\b([A-Z][A-Za-z0-9 &|]{2,80})\b", bullets[0])
                    company_name = cap_phrase.group(1).strip() if cap_phrase else "⚠️ Not Found"
                else:
                    company_name = "⚠️ Not Found"

        quadrants[q_label] = [company_name]

    return axis_labels, quadrants

# Button triggers GPT-4 generation
if st.button("🚀 Generate AI Matrix"):
    with st.spinner("Thinking with GPT-4..."):
        # Filter data to selected industry
        filtered = df[df['Matched Industries'].apply(lambda inds: selected_industry in inds)].copy()

        if filtered.empty:
            st.warning(f"No companies matched the industry: {selected_industry}")
        else:
            # Sample up to 10 companies
            subset = filtered.sample(n=min(10, len(filtered)))

            # Format company info with links and descriptions
            subset['Formatted'] = subset.apply(
                lambda row: f"[{row['Portco Name']}]({row['Company URL']}) – {row['Company Description']}",
                axis=1
            )
            company_info = "\n".join(subset['Formatted'].tolist())

            # Build the prompt
            prompt = f"""
You are a venture capital analyst. Your task is to generate a 2x2 matrix analysis for the selected industry: **{selected_industry}**.

🧭 INSTRUCTIONS:
1. Based on the selected sector from the dropdown menu, identify all the companies that fall within that industry. These companies may contain the selected industry in any one of their industry columns.
2. Analyze the company descriptions to identify common themes, trends, or strategic insights that can be used to create a 2x2 matrix with for uniques segments.
3. Carefully read the company names, descriptions, and website links in the company list. Use them to identify common themes, technologies, market gaps, and positioning opportunities that can inform a 2x2 framework.
4. Carefully study EXAMPLE 1 and EXAMPLE 2. Emulate their tone, quadrant naming, description style, and formatting. These are reference models for what high-quality output looks like.
5. Create a 2x2 matrix using the format of an x/y plot with four directional intersections: x<>y, y<>-x, -x<>-y, and -y<>x. **THIS IS VERY IMPORTANT THAT THE INTERSECTION STRUCTURE IS FOLLOWED.**
6. Ensure that x and -x, and y and -y represent meaningful opposites or tensions within the selected industry.
7. Each quadrant should include:
   - A **bolded title** in the format: **[Segment A] <> [Segment B]: [Intersection Title]**
   - A short (1–2 sentence) description of the strategic importance of the intersection
   - One concise sentence explaining why this intersection matters
   - One company from the list that best fits the intersection
   - A **bullet-pointed** explanation, written in **narrative style** (like EXAMPLE 1), with the **company name hyperlinked** to its website (provided in the company list). **Do not use “Company Example: ...” format**

8. Before the intersections, include this subtitle (on its own line, under the title):  
   *Supported by [Hustle Fund’s Current Portfolio](https://www.hustlefund.vc/founders)*

9. Include an **Introduction** section (2 sentences) that summarizes key industry trends.  
   Add a horizontal rule using `---` after the intro.

10. After the intersections, include a **Conclusion** section:
   - Reflect on how these intersections build a complete understanding of the industry
   - Include a paragraph comparing the differences across intersections
   - Add a horizontal rule before this section too

📌 Required Intersection Structure:
Output the intersections in this order:

1. x <> y  
2. y <> -x  
3. -x <> -y  
4. -y <> x  

- You MUST use four **unique intersections**, each with a distinct combination of segment A and segment B. So 4 intersections total!
- These must represent meaningful, opposing forces or concepts. Ensure that "x" and "-x" are thematic opposites, and so are "y" and "-y". Do not repeat segments. This structure is required.
- Use each segment only **once** on each side of a pairing
- Not repeat a segment pairing in any form (e.g., `A <> B` is the same as `B <> A`)
- Not repeat the same segment in more than one pairing

📌 Introduction:

1. The bold title: **[Industry Name] AI Supported Deep Dive**
2. A 21-2 sentence narrative-style intro paragraph on macro trends and tensions
4. Then, on a new line, add the italic note:  
   *Supported by [Hustle Fund’s Current Portfolio](https://www.hustlefund.vc/founders)*
5. Insert a horizontal rule using `---` after this block.

- Before listing the intersections, write a short 2-sentence paragraph introducing the macro trends, tensions, or themes that emerged from analyzing the companies. Frame the 2x2 matrix as a lens to explore these.
- Then insert a horizontal rule using `---`.
- Write a 1-2 sentence introductory paragraph with a narrative tone. Begin by describing how the selected industry is evolving due to technological innovation, shifting demands, or structural inefficiencies. Then, explain why it's important to "zoom in" on key intersections to better understand the market dynamics and opportunities.
- Use language similar to EXAMPLE 2 (healthcare): thoughtful, forward-looking, and high-level. This introduction should frame the matrix as a lens through which we explore deeper tensions and innovations shaping the industry.

📌 Segment Intersections:
Write all 4 intersections in the required format.

✍️ Style Guide:
- Carefully study the writing style of EXAMPLE 1 and EXAMPLE 2 below. Your tone should mirror their clarity, rhythm, and narrative flow.
- Avoid robotic phrasing like “Companies in this quadrant...” or “This quadrant represents...”
- Write confidently, like an investor explaining market dynamics to their partners.
- Focus on the tension or interplay between segments — not just restating them.
- Bold each intersection header.
- Write company descriptions as bullet points `-`, not in “Company Example: …” format.
- Integrate the company name into the narrative with a Markdown hyperlink.

📌 Output Format:
**[Segment A] <> [Segment B]: [Intersection Title]**

[1 sentence description of the strategic importance of the intersection]

- introducting the company in a fluid matter
- [Hyperlinked Company Name] [Natural, flowing description of the company and why it fits here]

- Here is an example for you to reference from example_one.txt: "Anticipating and Mitigating Risk: In a world of constant flux, understanding multi-tier supply chain risks is paramount. Ceres Technology is at the forefront, leveraging an AI-powered platform and over 25,000 real-time datasets to help businesses predict and preempt disruptions months in advance."

(Repeat this format for all 4 intersections)

📌 Conclusion:
After listing all intersections, write a final 1-2 sentence paragraph reflecting on how the intersections combine to give a deeper understanding of the industry as a whole.  
Compare the differences between intersections and emphasize how they work together.  
Insert a horizontal rule before this section.

---

--- EXAMPLE 1 ---
{example_one}

--- EXAMPLE 2 ---
{example_two}

Company list:
{company_info}
"""

            # Make the GPT-4 call
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            # Display the result
            gpt_output = response.choices[0].message.content

            # Split the output before first intersection header
            import re
            split_point = re.search(r"\*\*.*<>.*:.*\*\*", gpt_output)
            if split_point:
                intro_text = gpt_output[:split_point.start()].strip()
                body_text = gpt_output[split_point.start():].strip()
            else:
                intro_text = gpt_output
                body_text = ""

            # 1️⃣ Show the intro
            st.markdown(intro_text)

            # 2️⃣ Show chart using example segments & companies (replace with dynamic if needed)

            # 2️⃣ Extract dynamic axis & quadrant data from GPT output
            axis_labels, quadrants = extract_labels_and_quadrants(gpt_output)
            if axis_labels:
                fig = plot_quadrant_chart(axis_labels, quadrants)
                st.pyplot(fig)
            else:
                st.warning("Could not parse quadrant data from GPT response.")

            # 3️⃣ Show the rest of the GPT output (intersections + conclusion)
            if body_text:
                st.markdown(body_text)
        
#streamlit run matrix_interface.py