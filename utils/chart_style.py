def apply_chart_style(fig, tickangle=0):

    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        ),
        xaxis_tickangle=tickangle
    )

    return fig