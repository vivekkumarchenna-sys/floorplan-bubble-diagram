"""Module M2: geometry-based typed connectivity graph construction.

Builds the typed room-adjacency graph - door / open passage / shared wall -
directly from a semantic segmentation mask. The SAME procedure is applied to
the ground-truth mask (yielding the geometry-based reference of Section 4.2)
and to the predicted mask (yielding the pipeline graph); see Section 5.2 and
Appendix D.1-D.2. No edge weights, no pruning: every room pair is exactly one
of door / open passage / shared wall, or unconnected.
"""
import numpy as np
from scipy import ndimage
from build_graph import CLASS_NAMES

from build_graph import CLASS_NAMES

def rooms_of(mask):
    R={}; rid=0
    for c in range(1,10):
        lab,n=ndimage.label(mask==c)
        for k in range(1,n+1):
            cm=(lab==k)
            if cm.sum()<100: continue
            rid+=1; R[rid]=(c,cm)
    return R

def build_true_graph(mask, adj_dil=9, min_share=25, door_dil=8, min_wall_contact=250, door_frac=0.4, open_min=45):
    R=rooms_of(mask)
    wall=(mask==10)
    dils={i:ndimage.binary_dilation(cm,iterations=adj_dil) for i,(c,cm) in R.items()}
    # interior doors only (class 11), ignore front door (13)
    door=(mask==11); lab,n=ndimage.label(door)
    door_edges={}
    for k in range(1,n+1):
        dc=(lab==k)
        if dc.sum()<15: continue
        dd=ndimage.binary_dilation(dc,iterations=door_dil)
        touched=[(i,(dd&cm).sum()) for i,(c,cm) in R.items() if (dd&cm).sum()>20]
        if len(touched)<2: continue
        # A door opens from its hub room (the one it overlaps most - typically the
        # circulation space) onto every room it substantially reaches (overlap >=
        # door_frac of the hub overlap). Thin doors join a single pair; a wide
        # corner opening at a T-junction joins the hub to both rooms it opens onto.
        touched.sort(key=lambda t:-t[1])
        hub,hov=touched[0]
        for j,ov in touched[1:]:
            if ov>=door_frac*hov:
                a,b=sorted([hub,j]); door_edges[(a,b)]='door'
    # remaining adjacencies: OPEN PASSAGE (rooms meet through a gap, no door) vs
    # SOLID SHARED-WALL (separated by an unbroken wall). A gap where the two room
    # regions come directly together (within ~3 px, no wall between) is an opening.
    edges=dict(door_edges)
    ids=list(R)
    tight={i:ndimage.binary_dilation(cm,iterations=3) for i,(c,cm) in R.items()}
    for ii in range(len(ids)):
        for jj in range(ii+1,len(ids)):
            a,b=ids[ii],ids[jj]
            if (a,b) in edges: continue
            zone=dils[a]&dils[b]
            if zone.sum()<min_share: continue
            wallc=(zone&wall).sum()
            opening=(tight[a]&R[b][1]).sum()     # room-b floor within 3 px of room-a floor
            if opening>=open_min:
                edges[(a,b)]='open'
            elif wallc>=min_wall_contact:
                edges[(a,b)]='shared-wall'
    return R, edges

def name_map(R, mask):
    # human names by position
    byclass={}
    for i,(c,cm) in R.items(): byclass.setdefault(c,[]).append((i,cm))
    names={}
    for c,lst in byclass.items():
        lst2=[]
        for i,cm in lst:
            ys,xs=np.where(cm); lst2.append((i,xs.mean(),ys.mean()))
        lst2.sort(key=lambda t:(t[1] if c==1 else t[2]))
        for idx,(i,cx,cy) in enumerate(lst2,1):
            nm=CLASS_NAMES[c]
            hint={('Bedroom',1):'Bedroom1',('Bedroom',2):'Bedroom2',('Bathroom',1):'Bathroom1',('Bathroom',2):'Bathroom2'}
            names[i]=hint.get((nm,idx), nm if len(lst2)==1 else f'{nm}{idx}')
    return names
