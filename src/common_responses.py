import re

# Common question patterns and responses
COMMON_RESPONSES = {
    # Greetings
    r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b': 
        "Hello! I'm your Medical Assistant. How can I help you today? You can ask me about symptoms, medications, treatments, or general health information.",
    
    # Gratitude
    r'\b(thank you|thanks|thank u|thx)\b': 
        "You're welcome! Is there anything else I can help you with regarding your health?",
    
    # Goodbye
    r'\b(bye|goodbye|see you|exit|quit)\b': 
        "Take care! Remember to consult with a healthcare professional for personalized medical advice. Stay healthy!",
    
    # How are you
    r'\b(how are you|how r u|how are u)\b': 
        "I'm functioning well, thank you for asking! More importantly, how can I assist you with your health concerns today?",
    
    # What can you do
    r'\b(what can you do|what do you do|how can you help|capabilities)\b': 
        "I can help you with:\n• Understanding medical conditions and symptoms\n• Information about medications and treatments\n• General health advice\n• Answering medical questions based on trusted sources\n\nPlease note: I'm not a replacement for professional medical advice. What would you like to know?",
    
    # Who are you
    r'\b(who are you|what are you|tell me about yourself)\b': 
        "I'm a Medical AI Assistant powered by advanced language models and medical literature. I provide health information based on verified medical sources, but I'm not a substitute for professional medical consultation.",
    
    # Emergency
    r'\b(emergency|urgent|911|help me|dying|critical)\b': 
        "⚠️ If this is a medical emergency, please:\n• Call emergency services (911 or your local emergency number) immediately\n• Go to the nearest emergency room\n• Contact your doctor right away\n\nI can provide general information, but emergency situations require immediate professional medical attention!",
    
    # Appointment/Doctor
    r'\b(make appointment|book appointment|find doctor|need doctor)\b': 
        "I cannot schedule appointments, but I recommend:\n• Contacting your primary care physician directly\n• Using your healthcare provider's online portal\n• Calling your local hospital or clinic\n\nIs there any medical information I can help you with in the meantime?",
    
    # Prescription
    r'\b(prescribe|prescription|give me medicine|need medication)\b': 
        "I cannot prescribe medications. Only licensed healthcare providers can prescribe drugs. Please consult with:\n• Your primary care doctor\n• A specialist if needed\n• A telehealth service\n\nI can provide information about medications if you'd like to learn more.",
    
    # Diagnosis
    r'\b(diagnose me|what do i have|am i sick|do i have)\b': 
        "I cannot provide diagnoses. Proper diagnosis requires:\n• Physical examination by a healthcare provider\n• Medical history review\n• Appropriate tests and imaging\n\nI can help you understand symptoms and conditions, but please consult a doctor for accurate diagnosis. What symptoms are you experiencing?",
    
    # Price/Cost
    r'\b(how much|cost|price|afford|insurance)\b': 
        "I don't have information about medical costs, as they vary by:\n• Location and facility\n• Insurance coverage\n• Specific procedures\n\nPlease contact:\n• Your healthcare provider's billing department\n• Your insurance company\n• Hospital financial services\n\nIs there medical information I can help you with?",
}


def check_common_question(question):
    question_lower = question.lower().strip()
    
    for pattern, response in COMMON_RESPONSES.items():
        if re.search(pattern, question_lower, re.IGNORECASE):
            return response
    
    return None
