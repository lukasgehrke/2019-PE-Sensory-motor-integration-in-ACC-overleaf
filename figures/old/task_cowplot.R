library(ggplot2)
library(cowplot)
install.packages('magick')
library(magick)

fpath <- '/Users/lukasgehrke/Documents/bpn_work/publications/2019-PE-Sensory-motor-integration-in-ACC-overleaf'
exp <- paste(fpath, 'experiment.pdf', sep = "")
task <- paste(fpath, 'Task_mismatch.PNG', sep = "")

ggdraw() + 
  draw_image(exp, scale = 0.5)
