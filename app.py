import gradio as gr
from model import build_vocabs, load_models, generate

print("Loading vocabularies from train.txt ...")
src_vocab, tgt_vocab = build_vocabs('train.txt')
print(f"Source vocab: {len(src_vocab)} | Target vocab: {len(tgt_vocab)}")

print("Loading trained model weights ...")
encoder, decoder = load_models(src_vocab, tgt_vocab,
                               'hw5.encoder', 'hw5.decoder')
print("Ready.")


def build_mr(name, eat_type, food, price, rating, area, family, near):
    """Assemble the meaning-representation token list from form fields."""
    parts = []
    if name:      parts.append(f"name[{name}]")
    if eat_type:  parts.append(f"eatType[{eat_type}]")
    if food:      parts.append(f"food[{food}]")
    if price:     parts.append(f"priceRange[{price}]")
    if rating:    parts.append(f"customer rating[{rating}]")
    if area:      parts.append(f"area[{area}]")
    if family:    parts.append(f"familyFriendly[{family}]")
    if near:      parts.append(f"near[{near}]")
    return parts


def run(name, eat_type, food, price, rating, area, family, near):
    tokens = build_mr(name, eat_type, food, price, rating, area, family, near)
    if not tokens:
        return "Please fill in at least one field.", "", ""

    mr_string = ", ".join(tokens)
    words, unknown = generate(tokens, src_vocab, tgt_vocab, encoder, decoder)

    if not words:
        description = "(The model couldn't generate from these inputs - none matched the training vocabulary.)"
    else:
        # light cleanup of tokenized output
        text = " ".join(words)
        text = text.replace(" .", ".").replace(" ,", ",").replace(' `` ', ' "')
        description = text[0].upper() + text[1:] if text else text

    note = ""
    if unknown:
        note = ("Note: these values weren't in the training vocabulary and were "
                "ignored - try options closer to the E2E dataset: " + ", ".join(unknown))

    return description, mr_string, note


with gr.Blocks(title="Restaurant Description Generator") as demo:
    gr.Markdown("# Restaurant Description Generator")
    gr.Markdown(
        "A seq2seq (GRU encoder-decoder) model that turns structured restaurant "
        "attributes into a natural-language description. Built from CS 6320 HW5."
    )

    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Name", placeholder="The Eagle")
            eat_type = gr.Dropdown(
                ["", "restaurant", "pub", "coffee shop"], label="Eat Type", value="")
            food = gr.Dropdown(
                ["", "English", "French", "Italian", "Japanese", "Chinese",
                 "Indian", "Fast food"], label="Food", value="")
            price = gr.Dropdown(
                ["", "cheap", "moderate", "high", "less than £20",
                 "£20-25", "more than £30"], label="Price Range", value="")
        with gr.Column():
            rating = gr.Dropdown(
                ["", "low", "average", "high", "1 out of 5",
                 "3 out of 5", "5 out of 5"], label="Customer Rating", value="")
            area = gr.Dropdown(
                ["", "city centre", "riverside"], label="Area", value="")
            family = gr.Dropdown(
                ["", "yes", "no"], label="Family Friendly", value="")
            near = gr.Textbox(label="Near", placeholder="Burger King")

    btn = gr.Button("Generate Description", variant="primary")

    out_desc = gr.Textbox(label="Generated Description", lines=3)
    out_mr = gr.Textbox(label="Meaning Representation (model input)", lines=2)
    out_note = gr.Textbox(label="Notes", lines=2)

    btn.click(run,
              inputs=[name, eat_type, food, price, rating, area, family, near],
              outputs=[out_desc, out_mr, out_note])

    gr.Examples(
        examples=[
            ["The Eagle", "coffee shop", "French", "moderate", "3 out of 5",
             "riverside", "yes", "Burger King"],
            ["Blue Spice", "coffee shop", "", "less than £20", "average",
             "city centre", "no", "Avalon"],
            ["The Rice Boat", "", "Japanese", "", "5 out of 5",
             "riverside", "yes", ""],
        ],
        inputs=[name, eat_type, food, price, rating, area, family, near],
    )


if __name__ == "__main__":
    demo.launch()
