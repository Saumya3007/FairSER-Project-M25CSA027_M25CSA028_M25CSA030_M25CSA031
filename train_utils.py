import env  
import torch
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader):
    model.eval()
    preds, trues, langs, gends = [], [], [], []
    with torch.no_grad():
        for X, attn, y, lang, gend in loader:
            out    = model(X.to(DEVICE), attn.to(DEVICE))
            preds += out.argmax(1).cpu().tolist()
            trues += y.tolist()
            langs += list(lang)
            gends += list(gend)
    macro = f1_score(trues, preds, average="macro",    zero_division=0)
    wtd   = f1_score(trues, preds, average="weighted", zero_division=0)
    group_f1 = {}
    for vals, arr in [(["english", "hindi"], langs), (["male", "female", "unknown"], gends)]:
        for gv in vals:
            idx = [i for i, v in enumerate(arr) if v == gv]
            if idx:
                group_f1[gv] = round(f1_score(
                    [trues[i] for i in idx],
                    [preds[i] for i in idx],
                    average="macro", zero_division=0), 4)
    lg = round(abs(group_f1.get("english", 0) - group_f1.get("hindi",  0)), 4)
    gg = round(abs(group_f1.get("male",    0) - group_f1.get("female", 0)), 4)
    return macro, wtd, group_f1, lg, gg, preds, trues