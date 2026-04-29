import pandas as pd
import random
import tkinter as tk
from tkinter import ttk
import re
import json


reviews = {}

help_text = """Нейросеть (ну не прям)
---------------------
Что я понимаю:
  • Типы: огонь, вода, трава, электричество, псих, дракон
  • Характеристики: сильный, быстрый, живучий, умный
    • Примеры: 'сильный огонь', 'быстрый электрический', 'живучий вода'
  • Поиск по имени покемона
  • Поиск по диапазону
    • Примеры: 'атака от 100 до 200', 'speed from 10 to 20'
  • 'случайный' - любой покемон
  • 'выход' - завершить
"""


def load_reviews():
    global reviews
    try:
        with open("reviews.json", "r", encoding="utf-8") as f:
            reviews = json.load(f)
    except:
        reviews = {}
        

def save_reviews():
    with open("reviews.json", "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)


def get_stats(name):
    name = name.lower()
    if name not in reviews or len(reviews[name]) == 0:
        return "Нет отзывов"

    ratings = [r["rating"] for r in reviews[name]]
    avg = sum(ratings) / len(ratings)

    return f"{avg:.1f} ({len(ratings)} отзывов)"


stats = {
        'атака': 'attack',
        'attack': 'attack',
        'speed': 'speed',
        'скорость': 'speed',
        'hp': 'hp',
        'здоровье': 'hp',
        'sp_attack': 'sp_attack',
        'спец': 'sp_attack'
    }


def get_numbers(query: str):
    stat = None
    
    for key in stats:
        if key in query:
            stat = stats[key]
            break

    if not stat:
        return None
    
    numbers = list(map(int, re.findall(r'\d+', query)))

    if len(numbers) >= 2:
        return stat, numbers[0], numbers[1]

    return None



df = pd.read_csv("Pokemon.csv")

load_reviews()


def simple_neural_network(query: str, top_k: int = 5, best: bool = True):
    query = query.lower()
    scores = []

    if len(query) < 3 or query in ['покемон', 'pokemon', 'кого', 'выбери']:
        for idx, row in df.iterrows():
            score = row['attack'] * 0.2 + row['speed'] * 0.2 + row['hp'] * 0.1 + random.uniform(0, 10)
            scores.append((row['name'], score))
    else:
        for idx, row in df.iterrows():
            score = 0

            if any(word in query for word in ['огонь', 'fire', 'огненный']):
                if row['type1'] == 'fire' or row['type2'] == 'fire':
                    score += 40
            if any(word in query for word in ['вода', 'water', 'водный']):
                if row['type1'] == 'water' or row['type2'] == 'water':
                    score += 40
            if any(word in query for word in ['трава', 'grass', 'травяной']):
                if row['type1'] == 'grass' or row['type2'] == 'grass':
                    score += 40
            if any(word in query for word in ['электричество', 'electric', 'электрический']):
                if row['type1'] == 'electric' or row['type2'] == 'electric':
                    score += 40
            if any(word in query for word in ['псих', 'psychic', 'психический']):
                if row['type1'] == 'psychic' or row['type2'] == 'psychic':
                    score += 40
            if any(word in query for word in ['дракон', 'dragon']):
                if row['type1'] == 'dragon' or row['type2'] == 'dragon':
                    score += 50

            if any(word in query for word in ['сильный', 'strong', 'мощный']):
                score += row['attack'] * 0.4 + row['sp_attack'] * 0.2
            if any(word in query for word in ['быстрый', 'fast', 'скоростной']):
                score += row['speed'] * 0.6
            if any(word in query for word in ['живучий', 'tank', 'выносливый']):
                score += row['hp'] * 0.4 + row['defense'] * 0.2 + row['sp_defense'] * 0.1
            if any(word in query for word in ['умный', 'smart', 'интеллектуальный']):
                score += row['sp_attack'] * 0.5

            if score == 0:
                score = row['attack'] * 0.1 + row['speed'] * 0.1 + row['hp'] * 0.05

            score += random.uniform(0, 5)
            scores.append((row['name'], score))

    scores.sort(key=lambda x: x[1], reverse=best)
    return scores[:top_k]


def search_by_range(stat, min_val, max_val, top_k=5, best=True):
    filtered = df[(df[stat] >= min_val) & (df[stat] <= max_val)].copy()

    if filtered.empty:
        return []

    filtered = filtered.sort_values(by=stat, ascending=not best)

    results = []
    for _, row in filtered.head(top_k).iterrows():
        score = row[stat]
        results.append((row['name'], score))

    return results


def format_pokemon_output(results):
    output = ""
    for i, (name, score) in enumerate(results, 1):
        pokemon = df[df['name'] == name].iloc[0]
        type2 = pokemon['type2'] if pd.notna(pokemon['type2']) else ''

        output += f"{i}. {name.upper()}\n"
        output += f"Тип: {pokemon['type1']} {type2}\n"
        output += f"Атака: {pokemon['attack']} | Скорость: {pokemon['speed']} | ХП: {pokemon['hp']} | Спец. атака: {pokemon['sp_attack']}\n"
        output += f"Совместимость: {score:.1f}\n"
        output += "-" * 40 + "\n"

    return output


def add_review():
    name = entry.get().strip().lower()
    text = review_entry.get().strip()

    try:
        rating = int(rating_entry.get())
        if rating < 1 or rating > 5:
            raise ValueError
    except:
        text_output.insert(tk.END, "\nОшибка: рейтинг 1-5\n")
        return

    if name == "":
        text_output.insert(tk.END, "\nВведите имя покемона\n")
        return

    if name not in reviews:
        reviews[name] = []

    reviews[name].append({
        "rating": rating,
        "text": text
    })

    save_reviews()

    review_entry.delete(0, tk.END)

    text_output.insert(tk.END, f"\n✅ Отзыв добавлен для {name.upper()}\n")


def handle_search():
    query = entry.get().strip().lower()
    best = best_var.get()

    if query in ['случайный', 'random']:
        pokemon = df.sample(1).iloc[0]
        type2 = pokemon['type2'] if pd.notna(pokemon['type2']) else ''

        output = f"СЛУЧАЙНЫЙ ПОКЕМОН\n\n"
        output += f"{pokemon['name'].upper()}\n"
        output += f"Тип: {pokemon['type1']} {type2}\n"
        output += f"Атака: {pokemon['attack']} | Скорость: {pokemon['speed']} | ХП: {pokemon['hp']} | Спец. атака: {pokemon['sp_attack']}\n"

        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, output)
        return

    matched = df[df['name'].str.lower() == query]

    if not matched.empty:
        pokemon = matched.iloc[0]
        stats = get_stats(pokemon['name'])
        type2 = pokemon['type2'] if pd.notna(pokemon['type2']) else ''

        output = f"{pokemon['name'].upper()}\n\n"
        output += f"Тип: {pokemon['type1']} {type2}\n"
        output += f"Атака: {pokemon['attack']} | Скорость: {pokemon['speed']} | ХП: {pokemon['hp']} | Спец. атака: {pokemon['sp_attack']}\n"
        output += f"{stats}\n"

        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, output)
        return
    
    range_query = get_numbers(query)

    if range_query:
        stat, min_val, max_val = range_query
        results = search_by_range(stat, min_val, max_val, best=best)

        if not results:
            text_output.delete(1.0, tk.END)
            text_output.insert(tk.END, "Ничего не найдено по диапазону")
            return

        output = ""
        for i, (name, score) in enumerate(results, 1):
            pokemon = df[df['name'] == name].iloc[0]
            type2 = pokemon['type2'] if pd.notna(pokemon['type2']) else ''

            output += f"{i}. {name.upper()}\n"
            output += f"Тип: {pokemon['type1']} {type2}\n"
            output += f"Атака: {pokemon['attack']} | Скорость: {pokemon['speed']} | ХП: {pokemon['hp']} | Спец. атака: {pokemon['sp_attack']}\n"
            output += "-" * 40 + "\n"

        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, output)
        return

    results = simple_neural_network(query, best=best)
    output = format_pokemon_output(results)

    text_output.delete(1.0, tk.END)
    text_output.insert(tk.END, output)


root = tk.Tk()
root.title("ПР9")
root.geometry("670x670")

text_output = tk.Text(root, height=25, wrap="word", font=("Consolas", 10))
text_output.pack(padx=10, pady=10, fill="both", expand=True)

text_output.insert(tk.END, help_text)

frame = tk.Frame(root)
frame.pack(padx=10, pady=5, fill="x")

review_frame = tk.Frame(root)
review_frame.pack(padx=10, pady=5, fill="x")

rating_entry = tk.Entry(review_frame, width=5)
rating_entry.pack(side="left", padx=5)
rating_entry.insert(0, "5")

review_entry = tk.Entry(review_frame)
review_entry.pack(side="left", fill="x", expand=True, padx=5)

add_review_btn = ttk.Button(review_frame, text="Оставить отзыв")
add_review_btn.pack(side="left", padx=5)

entry = tk.Entry(frame, font=("Arial", 12))
entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

add_review_btn.config(command=add_review)
root.bind('<Return>', lambda event: handle_search())

best_var = tk.BooleanVar(value=True)
best_check = tk.Checkbutton(frame, text="По убыванию", variable=best_var)
best_check.pack(side="left", padx=5)

btn = ttk.Button(frame, text="Найти", command=handle_search)
btn.pack(side="left")

root.mainloop()