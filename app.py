from flask import Flask, request , render_template
from src.pipeline.predict_pipiline import CustomData,PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict' , methods=['GET' , 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))         
        )

        data_df = data.get_data_as_df()
        predict_pipeline = PredictPipeline()
        res = predict_pipeline.predict(data_df)

        return render_template('form.html' , results = res[0])

if __name__ == '__main__':
    app.run(debug=True , host = '0.0.0.0' , port = 5000)