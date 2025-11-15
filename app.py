import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel, PeftConfig
import re

@st.cache_resource
def load_model():
    """Load the model exactly as in the notebook"""
    try:
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("./model")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Load LoRA adapter - use the same config as training
        model = PeftModel.from_pretrained(
            base_model, 
            "./model",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Set to evaluation mode
        model.eval()
        
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def generate_code(pseudo_code, model, tokenizer, max_length=512):
    """
    Generate code exactly as in the notebook training
    Uses the same prompt format and generation parameters
    """
    # Use the EXACT same prompt format from training
    PROMPT = f"Translate pseudo-code to C++:\n{pseudo_code}\n\nC++ Code:\n"
    
    # Format exactly as in training
    full_text = PROMPT + "<|endoftext|>"
    
    # Tokenize with same parameters as training
    inputs = tokenizer(
        full_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length,
        padding=False
    )
    
    # Generate with same parameters as in notebook
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,  # Match notebook temperature
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
        )
    
    # Decode and extract code
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the code part after "C++ Code:" (same logic as training)
    if "C++ Code:" in generated_text:
        code_part = generated_text.split("C++ Code:")[1].strip()
        # Remove everything after <|endoftext|> if present
        if "<|endoftext|>" in code_part:
            code_part = code_part.split("<|endoftext|>")[0].strip()
        return code_part
    else:
        return generated_text.replace(PROMPT, "").strip()

def test_generation(model, tokenizer):
    """Test with the same examples from training to verify consistency"""
    test_cases = [
        "Initialize sum to 0\nFor each number in list:\n    Add number to sum\nReturn sum",
        "Read input number\nIf number is even:\n    Print 'Even'\nElse:\n    Print 'Odd'",
        "Set result to 1\nFor i from 1 to n:\n    Multiply result by i\nReturn result"
    ]
    
    results = []
    for pseudo in test_cases:
        code = generate_code(pseudo, model, tokenizer)
        results.append({"pseudo": pseudo, "code": code})
    
    return results

def main():
    st.set_page_config(
        page_title="Pseudo-code to C++ Translator",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Pseudo-code to C++ Translator")
    st.markdown("Translate your pseudo-code into C++ using fine-tuned GPT-2 with LoRA")
    
    # Load model
    with st.spinner("Loading model... This may take a few seconds."):
        model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.error("Failed to load model. Please check if all model files are in the 'model' directory.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Model info
    with st.expander("Model Information"):
        st.write("**Base Model:** GPT-2")
        st.write("**Fine-tuning:** LoRA (Low-Rank Adaptation)")
        st.write("**Training Data:** SPOC Dataset")
        st.write("**Task:** Pseudo-code to C++ translation")
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Generation Parameters")
    
    max_length = st.sidebar.slider(
        "Max Length", 
        min_value=100, 
        max_value=1024, 
        value=512, 
        step=50,
        help="Maximum length of generated text"
    )
    
    temperature = st.sidebar.slider(
        "Temperature", 
        min_value=0.1, 
        max_value=1.5, 
        value=0.7, 
        step=0.1,
        help="Higher values = more creative, Lower values = more focused"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**💡 Usage Tips:**\n"
        "- Use clear, structured pseudo-code\n"
        "- Be specific about variable names and operations\n"
        "- Include input/output descriptions\n"
        "- Test with the examples below first"
    )
    
    # Main input area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Pseudo-code")
        pseudo_code = st.text_area(
            "Enter your pseudo-code:",
            height=250,
            placeholder="Example:\n"
            "Initialize sum to 0\n"
            "For each number in numbers list:\n"
            "    Add number to sum\n"
            "Calculate average = sum / count of numbers\n"
            "Print average",
            key="pseudo_input"
        )
        
        # Quick actions
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            if st.button("🔄 Generate Code", type="primary", use_container_width=True):
                st.session_state.generate_clicked = True
                
        with col1_2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.pseudo_input = ""
                st.session_state.generate_clicked = False
                st.rerun()
    
    with col2:
        st.subheader("🔄 Generated C++ Code")
        
        if st.session_state.get('generate_clicked', False) and pseudo_code.strip():
            with st.spinner("Generating C++ code..."):
                try:
                    # Update generation with temperature
                    generated_code = generate_code_with_temp(
                        pseudo_code, 
                        model, 
                        tokenizer, 
                        max_length, 
                        temperature
                    )
                    
                    # Display code
                    st.code(generated_code, language="cpp", line_numbers=True)
                    
                    # Code metrics
                    lines = len(generated_code.split('\n'))
                    chars = len(generated_code)
                    st.caption(f"📊 Code stats: {lines} lines, {chars} characters")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download C++ Code",
                        data=generated_code,
                        file_name="generated_code.cpp",
                        mime="text/x-c++src",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error during code generation: {str(e)}")
        
        elif not pseudo_code.strip():
            st.info("👆 Enter pseudo-code on the left and click 'Generate Code'")
        else:
            st.info("👆 Click 'Generate Code' to translate your pseudo-code")

    # Examples section
    st.markdown("---")
    st.subheader("📚 Try These Examples")
    
    examples = {
        "Find Maximum": """Initialize max to first element
For each element in array:
    If element > max:
        Set max to element
Return max""",
        
        "Factorial": """Function factorial(n):
    If n == 0:
        Return 1
    Else:
        Return n * factorial(n-1)""",
        
        "Binary Search": """Function binary_search(arr, target):
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
        
        "Sum Array": """Initialize sum to 0
For each number in numbers:
    Add number to sum
Return sum""",
        
        "Check Prime": """Function is_prime(n):
    If n < 2:
        Return false
    For i from 2 to sqrt(n):
        If n % i == 0:
            Return false
    Return true"""
    }
    
    # Create columns for examples
    cols = st.columns(3)
    example_keys = list(examples.keys())
    
    for i, col in enumerate(cols):
        if i < len(example_keys):
            example_name = example_keys[i]
            with col:
                if st.button(f"🧩 {example_name}", use_container_width=True):
                    st.session_state.pseudo_input = examples[example_name]
                    st.session_state.generate_clicked = True
                    st.rerun()

def generate_code_with_temp(pseudo_code, model, tokenizer, max_length=512, temperature=0.7):
    """Generate code with temperature control"""
    PROMPT = f"Translate pseudo-code to C++:\n{pseudo_code}\n\nC++ Code:\n"
    full_text = PROMPT + "<|endoftext|>"
    
    inputs = tokenizer(
        full_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length,
        padding=False
    )
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            repetition_penalty=1.1,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "C++ Code:" in generated_text:
        code_part = generated_text.split("C++ Code:")[1].strip()
        if "<|endoftext|>" in code_part:
            code_part = code_part.split("<|endoftext|>")[0].strip()
        return code_part
    else:
        return generated_text.replace(PROMPT, "").strip()

if __name__ == "__main__":
    main()
