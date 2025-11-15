import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel, PeftConfig
import re

@st.cache_resource
def load_model():
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("./model")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, "./model")
    
    return model, tokenizer

def generate_code(pseudo_code, model, tokenizer, max_length=512, temperature=0.7):
    prompt = f"Translate pseudo-code to C++:\n{pseudo_code}\n\nC++ Code:\n"
    
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the code part after "C++ Code:"
    code_match = re.search(r"C\+\+ Code:\n(.*)", generated_text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    else:
        return generated_text.replace(prompt, "").strip()

def main():
    st.set_page_config(
        page_title="Pseudo-code to C++ Translator",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Pseudo-code to C++ Translator")
    st.markdown("Translate your pseudo-code into C++ using fine-tuned GPT-2 with LoRA")
    
    # Load model
    try:
        model, tokenizer = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return
    
    # Sidebar for parameters
    st.sidebar.header("Generation Parameters")
    max_length = st.sidebar.slider("Max Length", 100, 1024, 512, 50)
    temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Tips:**\n"
        "- Use clear, structured pseudo-code\n"
        "- Keep pseudo-code under 200 words for best results\n"
        "- Adjust temperature for more/less creative outputs"
    )
    
    # Main input area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Input Pseudo-code")
        pseudo_code = st.text_area(
            "Enter your pseudo-code:",
            height=300,
            placeholder="Example:\n"
            "Initialize sum to 0\n"
            "For each number in the list:\n"
            "    Add number to sum\n"
            "Calculate average as sum divided by list size\n"
            "Return average"
        )
    
    with col2:
        st.subheader("🔄 Generated C++ Code")
        
        if st.button("Generate Code", type="primary") or pseudo_code:
            if pseudo_code.strip():
                with st.spinner("Generating C++ code..."):
                    try:
                        generated_code = generate_code(
                            pseudo_code, 
                            model, 
                            tokenizer, 
                            max_length, 
                            temperature
                        )
                        
                        st.code(generated_code, language="cpp")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Code",
                            data=generated_code,
                            file_name="generated_code.cpp",
                            mime="text/x-c++src"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during generation: {e}")
            else:
                st.warning("Please enter some pseudo-code to generate C++ code.")
    
    # Examples section
    st.markdown("---")
    st.subheader("📚 Examples")
    
    examples = st.selectbox(
        "Try these examples:",
        [
            "Select an example",
            "Find maximum in array",
            "Calculate factorial",
            "Binary search",
            "Bubble sort"
        ]
    )
    
    example_pseudo = {
        "Find maximum in array": """Initialize max to first element
For each element in array:
    If element > max:
        Set max to element
Return max""",
        
        "Calculate factorial": """Function factorial(n):
    If n == 0:
        Return 1
    Else:
        Return n * factorial(n-1)""",
        
        "Binary search": """Function binary_search(arr, target):
    low = 0
    high = length(arr) - 1
    
    While low <= high:
        mid = (low + high) / 2
        If arr[mid] == target:
            Return mid
        Else if arr[mid] < target:
            low = mid + 1
        Else:
            high = mid - 1
    
    Return -1""",
        
        "Bubble sort": """Function bubble_sort(arr):
    n = length(arr)
    For i from 0 to n-1:
        For j from 0 to n-i-2:
            If arr[j] > arr[j+1]:
                Swap arr[j] and arr[j+1]
    Return arr"""
    }
    
    if examples != "Select an example":
        st.session_state.pseudo_code = example_pseudo[examples]
        st.rerun()

if __name__ == "__main__":
    main()
