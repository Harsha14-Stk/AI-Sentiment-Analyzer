from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the model cleanly. This model only outputs POSITIVE or NEGATIVE.
# To ensure speed, we spin it up once globally.
try:
    analyzer = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception as e:
    print(f"Error loading transformer pipeline: {e}")
    analyzer = None

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment_result = None
    input_text = ""

    if request.method == 'POST':
        input_text = request.form.get('text_to_analyze', '').strip()
        
        if input_text and analyzer:
            # Run model pipeline
            raw_result = analyzer(input_text)[0]
            
            # Format outputs elegantly
            sentiment_result = {
                'label': raw_result['label'].capitalize(), # "Positive" or "Negative"
                'score': f"{raw_result['score'] * 100:.1f}%" # Convert to percentage e.g., 99.8%
            }

    return render_template('index.html', sentiment=sentiment_result, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)