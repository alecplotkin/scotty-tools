import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scrtt.plotting._flowplot import plot_flows


def main():
    df = pd.DataFrame({
        'source': 3*['A'] + 3*['B'] + 3*['C'],
        'target': 3*['A', 'B', 'C'],
        'outflow': np.arange(9) + 1,
        'inflow': 0.8 * np.arange(9)[::-1] + 1,
    })
    plot_flows(
        sources=df['source'],
        targets=df['target'],
        outflows=df['outflow'],
        inflows=df['inflow'],
        # palette={'A': 'r', 'B': 'b', 'C': 'g'},
        group_order=['C', 'A', 'B', ],
        kernel_width=0.5,
        flow_alpha=0.6,
        endpoint_linewidth=2,
        # endpoint_edgecolor=None,
    )
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
