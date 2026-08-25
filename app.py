import gradio as gr
from inference import predict

def run_model(current_backlog, planned_work, packers_assigned, bottleneck):

    if current_backlog is None:
        return "Enter Current Backlog", ""

    if planned_work is None:
        return "Enter Planned Work", ""

    if packers_assigned is None:
        return "Enter Packers Assigned", ""

    prediction, probability = predict(current_backlog, planned_work, packers_assigned, bottleneck)

    if probability is None:
        probability_display = "N/A"
    else:
        probability_display = f"{probability:.1%}"
    
    return prediction, probability

with gr.Blocks() as demo:
    gr.Markdown("# Next Hour Backlog Risk in a Fulfillment Center")
    gr.Markdown(
        "Provides a risk prediction based on planned work, current backlog, "
        "packers assigned, and whether a backlog is present."
    )

    planned_work = gr.Number(
        label="Enter Next Hour's Planned Work.",
        minimum=0,
        precision=0,
        placeholder="Enter number here..."
    )

    current_backlog = gr.Number(
        label="Enter Current Backlog present entering into next hour.",
        minimum=0,
        precision=0,
        placeholder="Enter number here..."
    )

    packers_assigned = gr.Number(
        label="Enter Number of Packers assigned for next hour.",
        minimum=0,
        precision=0,
        placeholder="Enter number here..."
    )

    bottleneck = gr.Radio(
        label="Is there a bottleneck present?",
        choices=[
            ("No", 0),
            ("Yes", 1)
        ],
        value=0
    )

    predict_btn = gr.Button("Predict Backlog Risk")

    pred_label = gr.Textbox(label="Predicted Backlog Risk")
    pred_conf = gr.Textbox(label="Probability of Backlog Risk")

    predict_btn.click(
        fn=run_model,
        inputs=[current_backlog, planned_work, packers_assigned, bottleneck],
        outputs=[pred_label, pred_conf]
    )

if __name__ == "__main__":
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
