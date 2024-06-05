import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv("results_unfreezing.csv")

    # Map the 'backbone' column to unfreezing levels
    data["unfreezing_level"] = data["structure"].replace(
        {
            "unfreezetop1": 1,
            "unfreezetop3": 3,
            "unfreezetop5": 5,
            "unfreezetop7": 7,
            "unfreezetop9": 9,
            "freezeall": 0,
        }
    )

    filtered_data = data[["val_auc", "task", "unfreezing_level"]]

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=filtered_data,
        x="unfreezing_level",
        y="val_auc",
        hue="task",
        style="task",
        markers=True,
        dashes=False,
    )
    plt.title("Validation AUC by Number of Unfrozen Backbone Layers")
    plt.xlabel("Number of Backbone Layers Unfrozen")
    plt.ylabel("Validation AUC")
    plt.legend(title="Structure")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
