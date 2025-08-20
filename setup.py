"""
Setup script for AI Chat Agent project
This script helps set up the environment and download required models
"""

import os
import sys
import subprocess
import platform

def run_command(command):
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(command, shell=True, check=True,
                              capture_output=True, text=True)
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
   
    packages = [
        "streamlit>=1.28.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "PyPDF2>=3.0.0",
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0"
    ]
   
    for package in packages:
        if not run_command(f"pip install {package}"):
            return False
   
    return True

def download_models():
    """Pre-download required models to cache"""
    print("🤖 Pre-downloading AI models...")
   
    try:
        # Download sentence transformer model
        from sentence_transformers import SentenceTransformer
        print("Downloading sentence transformer model...")
        SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer model downloaded")
       
        # Download language model
        # from transformers import AutoTokenizer, AutoModelForCausalLM
        # print("Downloading language model...")
        # AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        # Note: Model will be downloaded on first use to save space
        print("✅ Language model prepared")
       
        return True
       
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "models",
        "logs",
        "uploads"
    ]
   
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
   
    # Check available memory
    import psutil
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
   
    if available_gb < 4:
        print(f"⚠️  Warning: Only {available_gb:.1f}GB RAM available. Recommended: 8GB+")
    else:
        print(f"✅ RAM: {available_gb:.1f}GB available")
   
    # Check disk space
    disk = psutil.disk_usage('.')
    free_gb = disk.free / (1024**3)
   
    if free_gb < 10:
        print(f"⚠️  Warning: Only {free_gb:.1f}GB disk space available. Recommended: 10GB+")
    else:
        print(f"✅ Disk space: {free_gb:.1f}GB available")

def setup_environment_variables():
    """Setup environment variables"""
    print("🔧 Setting up environment...")
   
    # Create .env file template
    env_template = """# AI Chat Agent Environment Variables
#
# EXA API Key for web search functionality
# Get your key from: https://exa.ai
EXA_API_KEY="14f915e9-2fcb-4b59-b839-c027be34d922"

# Optional: HuggingFace token for some models
# HF_TOKEN=your_huggingface_token_here
"""
   
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_template)
        print("✅ Created .env template file")
    else:
        print("✅ .env file already exists")

def main():
    """Main setup function"""
    print("🚀 Setting up AI Chat Agent...")
    print("=" * 50)
   
    # Check Python version
    if not check_python_version():
        sys.exit(1)
   
    # Check system requirements
    try:
        import psutil
        check_system_requirements()
    except ImportError:
        print("Installing psutil for system checks...")
        run_command("pip install psutil")
        import psutil
        check_system_requirements()
   
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
   
    # Create directories
    create_directories()
   
    # Setup environment
    setup_environment_variables()
   
    # Download models
    if not download_models():
        print("⚠️  Model download failed, but you can continue. Models will download on first use.")
   
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your EXA API key (optional)")
    print("2. Run: streamlit run app.py")
    print("3. Open your browser to http://localhost:8501")
    print("\n🔧 For evaluation:")
    print("   python evaluate_20Q.py -N 100")
    print("\n📚 Read README.md for detailed usage instructions")

if __name__ == "__main__":
    main()