from flask import Flask, request, render_template_string
from typing import Any, List
import pandas as pd
import pickle 

app = Flask(__name__)

books_df = pd.read_csv('Data/Clean_book_data.csv')
with open('Models/knn_model_20.pkl', 'rb') as file:
    model = pickle.load(file)



def make_recommendation(model: Any, df: pd.DataFrame, book_title_or_isbn: str, top_n: int=5)-> List[str]:
    """
    Makes book recommendations based on a given book title or ISBN, excluding the input book.

    Args:
    model (Pipeline): The trained KNN pipeline model.
    df (pd.DataFrame): The DataFrame containing the book data.
    book_title_or_isbn (str): The book title or ISBN to base recommendations on.
    top_n (int): The number of recommendations to return.

    Returns:
    list: A list of recommended book titles, excluding the input book.
    """
    book = df[(df['Book-Title'].str.contains(book_title_or_isbn, na=False, case=False))]
    if book.empty:
        return "Book not found."
    index = book.index[0]
    preprocessed_book_features = model.named_steps['preprocessor'].transform(df[df.index == index])
    _, indices = model.named_steps['classifier'].kneighbors(preprocessed_book_features, n_neighbors=top_n+1)
    
    recommended_indices = indices[0]
    if index in recommended_indices:
        recommended_indices = recommended_indices[recommended_indices != index]
    
    recommendations = df.iloc[recommended_indices][:top_n]['Book-Title'].tolist()

    return recommendations


html_template = '''
    <form method="post">
        <label for="book">Enter a Book Title or ISBN:</label>
        <input type="text" id="book" name="book">
        <input type="submit" value="Get Recommendations">
    </form>
    {% if recommendations %}
        <h3>Recommendations:</h3>
        <ul>
            {% for rec in recommendations %}
                <li>{{ rec }}</li>
            {% endfor %}
        </ul>
    {% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        book_title_or_isbn = request.form['book']
        recommendations = make_recommendation(model, books_df, book_title_or_isbn, top_n=5)
    return render_template_string(html_template, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)