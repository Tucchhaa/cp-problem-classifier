import matplotlib.pyplot as plt
import pandas as pd
import os
import ast

FIGURES_DIR_NAME = "figures"

df = pd.read_csv('./dataset/problems.csv')
df["description_length"] = df["description"].astype(str).apply(len)
df["labels"] = df["labels"].apply(ast.literal_eval)
df["label_count"] = df["labels"].apply(len)

sources = df["source"].unique()

def save_plot(plot_name):
    file_name = f"{plot_name}.png"
    file_path = f"./{FIGURES_DIR_NAME}/{file_name}"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    plt.savefig(file_path)

    plt.close()

def plot_problems_count_per_source():
    plt.figure(figsize=(8, 6))

    source_counts = df['source'].value_counts()
    bars = plt.bar(source_counts.index, source_counts.values)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom'
        )

    plt.title('Number of Problems per Source')
    plt.xlabel('Source')
    plt.ylabel('Number of Problems')
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_plot("problems_count_per_source")

def plot_problems_per_description_length():
    def plot_for_df(source_df, source = None):
        if source is None:
            title = "Total Description Length Distribution"
            plot_name = "problems_per_description_length_total"
        else:
            title = f"Description Length Distribution for {source}"
            plot_name = f"problems_per_description_length_for_{source}"

        plt.figure()
        plt.hist(source_df["description_length"], bins='auto', edgecolor="black")
        plt.title(title)
        plt.xlabel("Description Length")
        plt.ylabel("Number of Problems")
        plt.grid(True)
        save_plot(plot_name)

    for source in sources:
        source_df = df[df["source"] == source]
        plot_for_df(source_df, source)

    plot_for_df(df)

def plot_problems_count_per_label():
    def plot_for_df(source_df, source = None):
        if source is None:
            title = "Total Label Distribution"
            plot_name = "problems_count_per_label_total"
        else:
            title = f"Label Distribution for {source}"
            plot_name = f"problems_count_per_label_for_{source}"

        label_counts = source_df["labels"].value_counts().sort_values(ascending=True)

        plt.figure(figsize=(10, 6))
        label_counts.plot(kind="bar")
        plt.title(title)
        plt.xlabel("Label")
        plt.ylabel("Number of Problems")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.grid(True)
        save_plot(plot_name)

    exploded_df = df.explode("labels")

    for source in sources:
        source_df = exploded_df[exploded_df["source"] == source]
        plot_for_df(source_df, source)

    plot_for_df(exploded_df)

def plot_labels_count_per_problem():
    def plot_for_df(source_df, source = None):
        if source is None:
            title = "Total Label Count Distribution"
            plot_name = "labels_count_per_problem_total"
        else:
            title = f"Label Count Distribution for {source}"
            plot_name = f"labels_count_per_problem_for_{source}"

        count_distribution = source_df["label_count"].value_counts().sort_index()

        plt.figure(figsize=(8, 5))
        count_distribution.plot(kind="bar")
        plt.title(title)
        plt.xlabel("Number of Labels")
        plt.ylabel("Number of Problems")
        plt.xticks(rotation=0)
        plt.grid(True)
        plt.tight_layout()
        save_plot(plot_name)

    for source in df["source"].unique():
        source_df = df[df["source"] == source]
        plot_for_df(source_df, source)

    plot_for_df(df)

def plot_figures():
    plot_problems_count_per_source()
    plot_problems_count_per_label()
    plot_problems_per_description_length()
    plot_labels_count_per_problem()
