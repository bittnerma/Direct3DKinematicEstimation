import numpy as np


def checkContinous(df, duration=5):
    discontinousFrameIdx = []
    previousFrame = 0
    frameList = df['frame'].tolist()
    groupIdx = []
    startGroupIdx = -1

    for i in range(len(df)):
        currentFrame = frameList[i]
        if currentFrame == 0:
            previousFrame = 0
            startGroupIdx += 1
            groupIdx.append(startGroupIdx)
            continue
        elif i < (len(df) - 1) and currentFrame != (previousFrame + duration) and currentFrame != (
                frameList[i + 1] - duration):
            discontinousFrameIdx.append(i)
            previousFrame = frameList[i + 1]
            startGroupIdx += 1
            continue
        elif i == (len(df) - 1) and currentFrame != (previousFrame + duration):
            discontinousFrameIdx.append(i)
            continue

        previousFrame = currentFrame
        groupIdx.append(startGroupIdx)

    df = df.drop(discontinousFrameIdx)

    df["groupIdx"] = groupIdx
    startFrame = df.groupby("groupIdx")["frame"].min()
    df["startFrame"] = 0
    groupIdx = np.array(groupIdx)
    for i in df["groupIdx"].unique().tolist():
        df.at[groupIdx == i, "startFrame"] = startFrame[i]

    return df
