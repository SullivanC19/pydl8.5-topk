from topk.globals import ALL_DATASETS
from topk.experiments.fig2.runner import main as run_fig2
from topk.experiments.fig3.runner import main as run_fig3
from topk.experiments.fig4.runner import main as run_fig4
from topk.experiments.fig6.runner import main as run_fig6
from topk.experiments.fig2.plotter import main as plot_fig2
from topk.experiments.fig3.plotter import main as plot_fig3
from topk.experiments.fig4.plotter import main as plot_fig4
from topk.experiments.fig6.plotter import main as plot_fig6

if __name__ == "__main__":
  for dataset in ALL_DATASETS:
    run_fig2(dataset)
    run_fig3(dataset)
    run_fig4(dataset)
    run_fig6(dataset)

  plot_fig2()
  plot_fig3()
  plot_fig4()
  plot_fig6()