import wandb
api = wandb.Api()
run = api.run("/bioasq/yesno/runs/tkto1kk3")
print(run.history())