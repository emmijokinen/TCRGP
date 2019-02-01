import numpy as np
import tensorflow as tf
import gpflow
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.metrics import roc_curve as roc_curve
from sklearn.decomposition import PCA as PCA
from scipy.cluster.vq import kmeans2
import csv
import matplotlib.pyplot as plt

alphabet='ARNDCEQGHILKMFPSTWYV-'
# subsmatfromAA2 assumes the alphabet has length 21 and that the last character is for the gap.
# Consider this when changing the alphabet

# Handle sequence data and features 

def max_len(seq_list):
    """Returns the maximum sequence length within the given list"""
    lmax=0
    for seq in seq_list:
        lmax=max( lmax, len(seq) )
    return lmax

def tcrs2nums(tcrs):
    """Converts a list of (TCR) sequences to numbers. Each letter is changed to its index in alphabet"""
    tcrs_num=[]
    n=len(tcrs)
    for i in range(n):
        t=tcrs[i]
        nums=[]
        for j in range(len(t)):
            nums.append(alphabet.index(t[j]))
        tcrs_num.append(nums)
    return tcrs_num

def remove_starred(cdrs):
    """Returned cdrs are like the given cdrs, but cdrs with stars are replaced by an empty entry.
    Ikeep contains the locations of cdrs which did not contain stars."""
    Ikeep = np.ones((len(cdr1s),1),dtype=bool)
    for i in range(len(cdrs)):
        if '*' in cdrs:
            cdrs[i]=[]
            Ic[i]=False
    return cdrs, Ikeep

def add_gap(tcr,l_max,gap_char='-'):
    """Add gap to given TCR. Returned tcr will have length l_max.
    If there is an odd number of letters in the sequence, one more letter is placed in the beginning."""  
    l = len(tcr)
    if l<l_max:
        i_gap=np.int32(np.ceil(l/2))
        tcr = tcr[0:i_gap] + (gap_char*(l_max-l))+tcr[i_gap:l]
    return tcr

def align_gap(tcrs,l_max=None,gap_char='-'):
    """Align sequences by introducing a gap in the middle of the sequence.
    If there is an odd number of letters in the sequence, one more letter is placed in the beginning."""  
    if l_max == None:
        l_max = max_len(tcrs)
    else:
        assert (l_max >= max_len(tcrs)), "Given max length must be greater than or equal to the max lenght of given sequences, "+str(max(ls))
    
    tcrs_aligned=[]
    for tcr in tcrs:
        tcrs_aligned.append(add_gap(tcr,l_max,gap_char))
    return tcrs_aligned

def check_align_cdr3s(cdr3s,lmaxtrain):
    """Check cdr3s for too long sequences or sequences containing characters outside alphabet
    returns cdr3s_letter (proper cdr3s aligned, but improper sequences are left as they are)
            cdr3s_aligned (proper cdr3s aligned, places of improper sequences are left empty)
            Ikeep3 (locations of proper cdr3s)
    Here improper means sequences that are longer than those in the training data or contain
    characters outside the used alphabet."""
    lmaxtest=max_len(cdr3s)
    
    Ikeep3=np.ones((len(cdr3s),),dtype=bool)
    cdr3s_aligned=[]
    cdr3s_letter =[]
    if lmaxtest>lmaxtrain:
        print('Maximum length for test CDR3s is '+str(lmaxtest)+
              ', but the maximum length for the trained model is '+str(lmaxes[0])+'.')
        print('You need to train a new model with longer sequences to get predictions for all sequences.')   
        
    for i in range(len(cdr3s)):
        if len(cdr3s[i])>lmaxtrain or not all([ c in alphabet for c in cdr3s[i]]):
            Ikeep3[i]=False
            cdr3s_aligned.append([])
            cdr3s_letter.append(cdr3s[i])
        else:
            ca = add_gap(cdr3s[i],lmaxtrain)
            cdr3s_aligned.append(ca)
            cdr3s_letter.append(ca)
        
    return cdr3s_letter, cdr3s_aligned, Ikeep3

def clip_cdr3s(cdr3s,clip):
    """Clip amino acids from the ends of the given cdr3s, clip[0] from beginning and clip[1] from the end.
    Clipping should be done after the alignment."""
    for i in range(len(cdr3s)):
        cdr3s[i]=cdr3s[i][clip[0]:-clip[1]]
    return cdr3s

def subsmatFromAA2(identifier,data_file='data/aaindex2.txt'):
    """Retrieve a substitution matrix from AAindex2-file, scale it between 0 and 1, and include gap"""
    with open(data_file,'r') as f:
        for line in f:
            if identifier in line:
                break
        for line in f:
            if line[0] == 'M':
                split_line=line.replace(',',' ').split()
                rows=split_line[3]
                cols=split_line[6]
                break

        subsmat=np.zeros((21,21),dtype=np.float)
        i0=0
        for line in f:
            i=alphabet.find(rows[i0])
            vals=line.split()   
            for j0 in range(len(vals)):
                j=alphabet.find(cols[j0])
                subsmat[i,j]=vals[j0]
                subsmat[j,i]=vals[j0]
            i0+=1    
            if i0>=len(rows):
                break
        
    subsmat[:-1,:-1]+=np.abs(np.min(subsmat))+1
    subsmat[:-1,:-1]/=np.max(subsmat)
    subsmat[-1,-1]=np.min(np.diag(subsmat)[:-1])
    
    return subsmat    

def get_pcs(subsmat,d):
    """Get first d pca-components from the given substitution matrix (or other matrix)"""
    pca = PCA(d)
    pca.fit(subsmat)
    pc = pca.components_
    return pc

def encode_with_pc(seq_lists, lmaxes, pc):
    """ Encode the sequence lists (given as numbers), with the given pc components (or other features)
    lmaxes contains the maximum lengths of the given sequences. """
    d = pc.shape[0]
    X = np.zeros((len(seq_lists[0]),d*sum(lmaxes)))
    i_start, i_end = 0, 0
    for i in range(len(seq_lists)):
        Di = d*lmaxes[i]
        i_end += Di
        for j in range(len(seq_lists[i])):
            X[j,i_start:i_end] = np.transpose( np.reshape( np.transpose( pc[:,seq_lists[i][j]] ), (Di,1) ) )
        i_start=i_end
    return X

def file2dict(filename,key_fields,store_fields,delimiter='\t'):
    """Read file to a dictionary.
    key_fields: fields to be used as keys
    store_fields: fields to be saved as a list
    delimiter: delimiter used in the given file."""
    dictionary={}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile,delimiter=delimiter)    
        for row in reader:
            keys = [row[k] for k in key_fields]
            store= [row[s] for s in store_fields]
                
            sub_dict = dictionary
            for key in keys[:-1]:
                if key not in sub_dict: 
                    sub_dict[key] = {}
                sub_dict = sub_dict[key]
            key = keys[-1]
            if key not in sub_dict:
                sub_dict[key] = []
            sub_dict[key].append(store)

    return dictionary
    
def create_cdr_dict(alignment='imgt',species=['human']):
    """Creates a dictionary of the CDRs (1, 2, and 2.5) corresponding to each V-gene. 
    If alignment='imgt', the CDRs will be aligned according to imgt definitions.
    Dictionary has form cdrs12[organism][chain][V-gene] = [cdr1,cdr2,cdr2.5]"""
    
    cdrs_all=file2dict('data/alphabeta_db.tsv',key_fields=['organism','chain','region','id'],store_fields=['cdrs'])

    cdrs = {}
    for organism in species:
        cdrs[organism]={}

        for chain in 'AB':
            cdrs[organism][chain]={}
            for g in cdrs_all[organism][chain]['V']:
                if alignment is 'imgt':
                    c = cdrs_all[organism][chain]['V'][g][0][0].replace('.','-').split(';')[:-1]
                else:
                    c = cdrs_all[organism][chain]['V'][g][0][0].replace('.','').split(';')[:-1]
                cdrs[organism][chain][g]=c
                
    return cdrs

def correct_vgene(v, chain='B'):
    """Makes sure that the given v gene is in correct format,
    handles a different few formats."""
    if chain is 'B':
        v = v.replace('TCR','TR').replace('TRBV0','TRBV').replace('-0','-')
    elif chain is 'A':
        v = v.replace('TCR','TR').replace('TRAV0','TRAV').replace('-0','-')
    else:
        print("chain must be 'A' or 'B'. No corrections were made")       
    return v.split(';')[0] # If there are multiple genes separated by ';', only return first one

def split_v(v):
    """Given V-gene, returns that V-gene, its subgroup, name and allele.
    If a part cannot be retrieved from v, empty string will be returned for that part"""
    star = '*' in v
    if '-' in v:
        vs = v.split('-')
        subgroup = vs[0]
        if star:
            vs=vs[-1].split('*')
            name=vs[0]
            allele=vs[1]
        else:
            name=vs[1]
            allele=''
    elif star:
        vs = v.split('*')
        subgroup = vs[0]
        name = ''
        allele = vs[1]
    else:
        subgroup=v
        name=''
        allele=''
    return([v,subgroup,name,allele])


def create_minimal_v_cdr_list(organism='human',chain='B',cdrtypes=['cdr1','cdr2','cdr25']):
    """Create a list that determines minimal level of information (subgroup, name, allele) 
    needed to determine the wanted cdrtypes.
    Possible organism are human and mouse and possible chains are A and B."""
    cdrs = create_cdr_dict(species=[organism])
    v_list=[]
    c_list=[]
    for v in cdrs[organism][chain]:
        v_list.append(split_v(v))
        c_list.append(cdrs[organism][chain][v])

    v_array=np.asarray(v_list)
    c_array=np.asarray(c_list)
                              
    i_cs = []
    if 'cdr1' in cdrtypes:
        i_cs.append(0)
    if 'cdr2' in cdrtypes:
        i_cs.append(1)
    if 'cdr25' in cdrtypes:
        i_cs.append(2)

    vc_list=[]
    subgroups = np.unique(v_array[:,1])
    for sub in subgroups:
        Is = sub==v_array[:,1]    
        cs = c_array[Is,i_cs]
        c = ''.join(cs[0])
        if np.all(list(''.join(x)==c for x in cs)):
            vc_list.append([sub,'any','any',cs[0]])
        else:
            name_list=v_array[Is,2]
            names = np.unique(name_list)
            for name in names:
                In = name==name_list
                cs2 = cs[In]
                c = ''.join(cs2[0])
                if np.all(list(''.join(x)==c for x in cs2)):
                    vc_list.append([sub,name,'any',cs2[0]])
                else:        
                    alleles = v_array[Is,3][In]
                    for allele,c in zip(alleles,cs2):
                        vc_list.append([sub,name,allele,c])
    return vc_list

def extract_cdrs_minimal_v(vgenes, organism, chain, cdrtypes,correctVs=True):
    """Get requested cdrs (cdr_types) from the vgenes where possible. 
    If all requested cdrs could not be obtained from a vgenes, empty entries are returned in their place"""
    if correctVs:
        vgenes=[correct_vgene(v,chain) for v in vgenes]
    
    cdrs12 = create_cdr_dict(organism,chain)
    vc_list = create_minimal_v_cdr_list(organism,chain,cdrtypes)
    
    ics = []
    if 'cdr1' in cdrtypes:
        ics.appens(0)
    if 'cdr2' in cdrtypes:
        ics.appens(1)
    if 'cdr25' in cdrtypes:
        ics.appens(2)
    
    cdrs=[[],[],[]] 
    for v in vgenes:
        try:
            cs = cdrs12[organism][chain][v]
            for i in ics:
                cdrs[i].append(cs[i])
        except:
            vs=split_v(v)
            notFound=True
            for row in vc_list:
                if ((vs[1]==row[0]) and ((vs[2]==row[1]) or (row[1]=='any') or (vs[2]=='1' and row[1]==''))
                    and (vs[3]==row[2] or row[2]=='any')):
                    
                    for i in ics:
                        cdrs[i].append(row[3][i])
                    notFound=False
                    continue
            if notFound:
                for i in ics:
                    cdrs[i].append([])   
    return cdrs


def read_vs_cdr3s_epis_subs(datafile,va='va',vb='vb',cdr3a='cdr3a',cdr3b='cdr3b',epis='epitope',subs='subjects',delimiter=','):
    """Reads VA-genes, VB-genes, CDR3As, CDR3Bs from the given data file. 
    The columns are determined by va, vb, cdr3a, and cdr3b. Any of them can also be None, 
    if they are not required.
    returns a list of lists [vas, vbs, cdr3as, cdr3bs, epis, subs]. If a specifier was None, 
    the corresponding list is empty."""
    
    with open(datafile, newline='') as csvfile:
        header = csvfile.readline()
    fields=header.strip().split(delimiter)
    
    cols=[]
    names = [va,vb,cdr3a,cdr3b,epis,subs]
    i_names = []
    for i, name in zip(range(6), names):
        if name is not None:
            cols.append(fields.index(name))
            i_names.append(i)
        
    va_vb_3a_3b_ep_su =  [[] for i in range(6)]
    lists = np.loadtxt(datafile,dtype=str,delimiter=delimiter,comments=None,unpack=True,skiprows=1,usecols=cols)
    if len(cols)==1:
        va_vb_3a_3b_ep_su[i_names[0]]=lists
    else:
        for i_names, li in zip(i_names,lists):
            va_vb_3a_3b_ep_su[i_names] = li
    
    return va_vb_3a_3b_ep_su

def get_sequence_lists(organism,datafile,cdr_types,delim,clip3,clip,
                       va='va',vb='vb',cdr3a='cdr3a',cdr3b='cdr3b',epis='epitope',subs='subjects'):
    """Get epitopes, subjects, cdrs, and max lengths of cdrs. 
    V-genes should contain all information necessary to get the requested cdr types."""
    
    cdrAtypes=cdr_types[0]
    cdrBtypes=cdr_types[1]
    [vas,vbs,cdr3as,cdr3bs,epis,subs] = read_vs_cdr3s_epis_subs(datafile,va,vb,cdr3a,cdr3b,epis,subs,delim)
        
    cdrs = create_cdr_dict(species=[organism])
    seq_lists = []    
    if 'cdr3' in cdrAtypes:
        cdr3as = align_gap(cdr3as)
        if clip3:
            cdr3as = clip_cdr3s(cdr3as,clip)
        seq_lists.append(tcrs2nums(cdr3as))
    
    ics=[]
    for i,c in zip(range(3),['cdr1','cdr2','cdr25']):
        if c in cdrAtypes:
            ics.append(i)
            
    if len(ics) > 0:
        vas = [correct_vgene(v,'A') for v in vas]
        css=[[],[],[]]
        for v in vas:
            cs = cdrs['human']['A'][v]
            for i in ics:
                css[i].append(cs[i])
        for i in ics:
            seq_lists.append(tcrs2nums(css[i]))

    if 'cdr3' in cdrBtypes:
        cdr3bs = align_gap(cdr3bs)
        if clip3:
            cdr3bs = clip_cdr3s(cdr3bs,clip)
        seq_lists.append(tcrs2nums(cdr3bs))
                    
    ics=[]
    for i,c in zip(range(3),['cdr1','cdr2','cdr25']):
        if c in cdrBtypes:
            ics.append(i)
    if len(ics) > 0:
        vbs = [correct_vgene(v,'B') for v in vbs]
        css=[[],[],[]]
        for v in vbs:
            cs = cdrs['human']['B'][v]
            for i in ics:
                css[i].append(cs[i])
        for i in ics:
            seq_lists.append(tcrs2nums(css[i]))

    lmaxes = []
    for seqs in seq_lists:
        lmaxes.append(len(seqs[0]))
        
    return epis,subs,seq_lists,lmaxes

def get_subjects(organism,epi,epitopes,subjects,min_subjects=5, cv=5):
    """Get subject list for given epitope. Define new subjects if there are not enough.
    This function is primarily inteded to be used with loso"""
    I = epitopes==epi
    subjects_epi = subjects[I]
    subjects_u = np.unique(subjects_epi)
    
    l_epis = sum(I)
    n_subs = len(subjects_u)
        
    print(organism+' '+epi+': '+str(n_subs)+ ' subjects, '+str(l_epis)+' samples')

    ind1 = 0
    if n_subs < min_subjects:
        print('Not enough subjects. Using ' +str(cv)+ '-fold cross-validation')
        subjects_u=list(range(1,cv+1))
        subjects_epi = np.asarray((subjects_u*int(np.ceil(l_epis/cv)))[:l_epis])    
        l_s=cv

    return subjects_epi, l_epis, subjects_u, n_subs, I


# Plotting

def plot_aurocs_ths(y_list,p_list,epi='',thresholds=None,dpi=200):
    """plot AUROCs"""
    if thresholds is None:
        thresholds=[0.0, 0.05, 0.1, 0.2]
    
    f=plt.figure(figsize=(12,5),dpi=dpi)
    plt.subplot(121)
    
    aucscores, samples = [], []
    for i in range(len(y_list)):
        y = y_list[i]
        p = p_list[i]
        aucscores.append(roc_auc(y,p))
        samples.append(len(y))
        fprs,tprs,_ = roc_curve(y,p,pos_label=1)  
        plt.plot(np.concatenate([[0],fprs]),np.concatenate([[0],tprs]),linewidth=0.75)
    
    mean_auc    = np.mean(aucscores)
    mean_wt_auc = np.sum(np.expand_dims(aucscores,1)*np.expand_dims(samples,1))/sum(samples)
    
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    plt.axis('square')
    plt.xlabel('FPRS')
    plt.ylabel('TPRS')
    plt.title(epi+ '\n mean AUC: {:1.4f}, mean weighted AUC: {:1.4f}'.format(mean_auc,mean_wt_auc))

    plt.subplot(122)
    
    y_all=np.concatenate(y_list)
    p_all=np.concatenate(p_list)
    fprs,tprs,th = roc_curve(y_all,p_all,pos_label=1)
        
    l_auc, = plt.plot(fprs,tprs)
    l_th, = plt.plot(fprs,th,linewidth=0.75)
    legs = [l_auc, l_th]
    labels=['ROC','threshold']
    
    # Threshold that gives shortest distance to upper left corner
    i_best=np.argmin(np.sqrt( (1-tprs)**2 + fprs**2 ))
    plt.plot([fprs[i_best],fprs[i_best]],[tprs[i_best],th[i_best]],'k',linewidth=0.75)
    l_best, = plt.plot([fprs[i_best],fprs[i_best]],[tprs[i_best],th[i_best]],'k.',linewidth=0.75)
    legs.append(l_best)
    labels.append('Best threshold: {:1.4f}; FPRS={:1.4f}, TPRS={:1.4f} '.format(th[i_best],fprs[i_best],tprs[i_best]))
    
    for i in range(len(thresholds)):
        ind = len(fprs)-1 - np.where((np.flip(fprs,0)-thresholds[i])<=0)[0][0]
        while ind-1>=0 and tprs[ind]==tprs[ind-1]:
            ind-=1
        
        plt.plot([fprs[ind],fprs[ind]],[tprs[ind],th[ind]],'gray',linewidth=0.75)
        leg, = plt.plot([fprs[ind],fprs[ind]],[tprs[ind],th[ind]],'.',color='gray',linewidth=0.75)
        legs.append(leg)
        labels.append('Threshold: {:1.4f}; FPRS={:1.4f}, TPRS={:1.4f} '.format(th[ind],fprs[ind],tprs[ind]))
    
    plt.xlabel('FPRS')
    plt.ylabel('TPRS, threshold')
    plt.axis('square')
    plt.xlim([-0.025,1.025])
    plt.ylim([-0.025,1.025])
    plt.xticks(np.arange(0,1.01,0.1))
    plt.yticks(np.arange(0,1.01,0.1))

    plt.title(epi+' AUROC: {:1.4f}'.format(roc_auc(y_all,p_all)))
    plt.legend(handles=legs,labels=labels,loc=(1.1,0.6))   

    plt.show()
    
    return mean_auc, mean_wt_auc


# Construct kernels, select inducing points, train and load models

def construct_rbf_kernel(d,lmaxes,lengthscales=[1.0],kernvaris=[1.0]):
    """Contsructs a kernel as as sum of GPFlow RBF-kernels with given lengthscales and variances.
    d is the number of features used, and lmaxes is a list of the lengths of the CDRs used."""
    l = len(lmaxes)
    i_start=0
    Di=d*lmaxes[0]
    i_end = Di
    
    if len(kernvaris)==1 and l>1:
        kernvaris = kernvaris * l       
    if len(lengthscales)==1 and l>1:
        lengthscales = lengthscales * l
    
    kernel = gpflow.kernels.RBF(Di,active_dims=list(range(i_start,i_end)),lengthscales=lengthscales[0],variance=kernvaris[0])
    
    for i in range(1,l):
        i_start = i_end
        Di = d*lmaxes[i]
        i_end += Di
        kernel += gpflow.kernels.RBF(Di,active_dims=list(range(i_start,i_end)),lengthscales=lengthscales[i],variance=kernvaris[i])
    return kernel

def select_Z_mbs(nZ,mbs,XP_tr):
    """Select inducing point locations with kmeans from training data XP_tr, and  minibatch size.
    n_tr = number of training points.
    If nZ<1, there will be nZ * n_tr inducing points. Otherwise there will be nZ training points. 
    Same applies for the minibatch size mbs, except that if mbs=0 or mbs > n_tr, mbs is set to n_tr"""  
    n_tr = XP_tr.shape[0]
    if nZ < 1:
        nZ = int(np.ceil(nZ*n_tr))
    Z = kmeans2(XP_tr, nZ, minit='points')[0]
    if mbs == 0 or mbs > n_tr:
        mbs = n_tr # use all data
    elif mbs < 1:
        mbs = int(np.ceil(mbs*n_tr))
    return Z, mbs

def loso(datafile,organism,epi,pc,cdr_types=[[],['cdr3']],l=1.0,var=1.0,m_iters=5000,lr=0.005,nZ=0,mbs=0,clip3=False,clip=[3,2],
         min_subjects=5,cv=5,delim=',',va='va',vb='vb',cdr3a='cdr3a',cdr3b='cdr3b',subs='subject',epis='epitope'):
    """
    Leave-on-subject-out cross-validation with TCRGP
    datafile: delimeted file which contains columns Epitope, Subject, va, vb, cdr3a, cdr3b. If some of them are not
        required to get the requsted cdr types, they may be empty.
    organism: 'human' or 'mouse' 
    epi: name of the epitope
    pc: principal components or features for each amino acid.
    cdr_types: CDRs utilized by the model. list that contains list of CDR types for chain A and chain B. 
        possible CDR types are cdr1, cdr2, cdr25 and cdr3.
        
    l: initial length scale for kernel. Can also be a list where there is a separate lengthscale for each CDR
        in the following order: cdr3a, cdr1a, cdr2a, cdr25a, cdr3b, cdr1b, cdr2b, cdr25b
    var: initial variance (weight) for kernel. Same format as with l.
    m_iters: maximum number of iterations
    lr: learning rate
    nZ: number of inducing points to be used with SVGP(selected with kmeans). If zero, VGP will be used.
    mbs: minibatch size, in case SVGP is used.
    clip3: bool, if True, clip amino acids from CDR3s as specified by clip
    clip: list, remove clip[0] amino acids from beginning and clip[1] amino acids from the end
    min_subjects: minimum number of subjects required for loso-cv. If there are less subjects, 
        do cv-fold cross-validation instead.
    cv: how many fold cross-cross validation in case loso is not possible.
    va,vb,cdr3a,cdr3b,sub,epis: names for the columns that contain information for VA-genes, VB-genes, CDR3As, CDR3Bs,
        subjects, and epitopes. Any of them can be None, if they are not required to get the requested cdr_types 
    returns mean AUC, mean weighted AUC, class lists for all subjects/folds, predictions for all subjects/folds
    """
    
    # Read data file and extract requested CDRs
    epitopes,subjects,cdr_lists,lmaxes = get_sequence_lists(organism,datafile,cdr_types,delim,clip3,clip,va,vb,cdr3a,cdr3b,epis,subs)

    # encode with pc components
    d = pc.shape[0]
    X = encode_with_pc(cdr_lists,lmaxes,pc)
    
    # Handle subject list that determins the training folds.
    subjects_epi, l_epi, subjects_u, n_subs, Ipos = get_subjects(organism,epi,epitopes,subjects,min_subjects,cv)
    
    Ineg = ~Ipos
    y = np.zeros((len(epitopes),1), dtype=int)
    y[Ipos] = 1
      
    y_list, p_list = [], []
    for subject in subjects_u:
        Isub = np.ones((l_epi),dtype=bool)
        Isub[subjects_epi==subject] = False
        
        I = np.ones((2*l_epi),dtype=bool)
        I[Ipos]= Isub
        I[Ineg]= Isub
        
        with tf.Session(graph=tf.Graph()):
            kernel = construct_rbf_kernel(d,lmaxes)
            if nZ == 0: # use VGP
                m = gpflow.models.VGP(X[I,:],y[I],kernel,gpflow.likelihoods.Bernoulli())
            else: # use SVGP
                # inducing locations by kmeans
                Z, mbs = select_Z_mbs(nZ,mbs,X[I,:])
                m = gpflow.models.SVGP(X[I,:],y[I],kernel,gpflow.likelihoods.Bernoulli(),Z=Z,minibatch_size=mbs)
            m.likelihood.variance = 1.0

            gpflow.train.AdamOptimizer(lr).minimize(m, maxiter=m_iters)
            p, _ = m.predict_y(X[~I,:])

        y_list.append(y[~I])
        p_list.append(p)
    
    mean_auc, mean_wt_auc = plot_aurocs_ths(y_list,p_list,epi)
    return [mean_auc, mean_wt_auc, y_list, p_list]

def train_classifier(datafile,organism,epi,pc,cdr_types=[[],['cdr3']],m_iters=5000,lr=0.005,nZ=0,mbs=0,l3_max=0,
                     clip3=False,clip=[3,2],delimiter=',',va='va',vb='vb',cdr3a='cdr3a',cdr3b='cdr3b',epis='epitope'):
    """
    Train classifier with TCRGP. Returns training AUC and parameters required for the rebuilding of the model.
    datafile: delimeted file which contains columns Epitope, Subject, va, vb, cdr3a, cdr3b. If some of them are not
        required to get the requsted cdr types, they may be empty.
    organism: 'human' or 'mouse' 
    epi: name of the epitope
    pc: principal components or features for each amino acid.
    cdr_types: CDRs utilized by the model. list that contains list of CDR types for chain A and chain B. 
        possible CDR types are cdr1, cdr2, cdr25 and cdr3.
        
    l: initial length scale for kernel. Can also be a list where there is a separate lengthscale for each CDR
        in the following order: cdr3a, cdr1a, cdr2a, cdr25a, cdr3b, cdr1b, cdr2b, cdr25b
    var: initial variance (weight) for kernel. Same format as with l.
    m_iters: maximum number of iterations
    lr: learning rate
    nZ: number of inducing points to be used with SVGP(selected with kmeans). If zero, VGP will be used.
    mbs: minibatch size, in case SVGP is used.
    clip3: bool, if True, clip amino acids from CDR3s as specified by clip
    clip: list, remove clip[0] amino acids from beginning and clip[1] amino acids from the end
    delimiter: delimiter used in datafile
    va,vb,cdr3a,cdr3b,epis: names for the columns that contain information for VA-genes, VB-genes, CDR3As, CDR3Bs,
        and epitopes. Any of them can be None, if they are not required to get the requested cdr_types 
    returns mean AUC, mean weighted AUC, class lists for all subjects/folds, predictions for all subjects/folds
    """
     
    # Read data file and extract requested CDRs
    epitopes,_,cdr_lists,lmaxes = get_sequence_lists(organism,datafile,cdr_types,delimiter,clip3,clip,va,vb,cdr3a,cdr3b,epis,subs=None)
    
    # encode with pc components
    d = pc.shape[0]
    X = encode_with_pc(cdr_lists,lmaxes,pc)
    
    # Class labels
    y = np.zeros((len(epitopes),1),dtype=int)
    y[epitopes==epi] = 1
    assert any(y),"epi did not occur in the training data. There we no positive samples."

    with tf.Session(graph=tf.Graph()):
        kernel = construct_rbf_kernel(d,lmaxes)
        if nZ == 0:
            m = gpflow.models.VGP(X,y,kernel,gpflow.likelihoods.Bernoulli())
        else:
            # inducing locations by kmeans
            Z, mbs = select_Z_mbs(nZ,mbs,X)
            m = gpflow.models.SVGP(X,y,kernel,gpflow.likelihoods.Bernoulli(),Z=Z,minibatch_size=mbs)
        m.likelihood.variance = 1.0

        gpflow.train.AdamOptimizer(lr).minimize(m, maxiter=m_iters)
        
        ls, vs = [], []
        if len(cdr_lists)>1:
            for i in range(len(cdr_lists)):
                ls.append(m.kern.kernels[i].lengthscales.value)
                vs.append(m.kern.kernels[i].variance.value)
        else:
            ls.append(m.kern.lengthscales.value)
            vs.append(m.kern.variance.value)

        param_list = [cdr_types,lmaxes,ls, vs, m.q_mu.value, m.q_sqrt.value, pc, y, X]

        if nZ>0:
            param_list.append(m.Z.value)
            param_list.append(mbs)
        else:
            param_list.append([])
            param_list.append([])
            
        p, _ = m.predict_y(X)
    
    auc = roc_auc(y,p) # Training AUC
    # param_list: [lmaxes,lengthscales,variance,q_mu,q_sqrt, pc,y, X (,Z,mbs)]
    return auc, param_list

def predict(filename, params, organism='human',va=None,vb=None,cdr3a=None,cdr3b='cdr3b',delimiter=','):
    """Do predictions for the TCRs in the given file. 
    Predictions are returned in the same order as the TCRs in the file.
    filename: name of the file containing the TCRs for testing
    params: parameters needed to rebuild the classification model. 
        This is can be obtained from the train_classifier-function
    cdr_types    
        """
   
    # Extract parameters
    [cdr_types,lmaxes,lengthscales,variances,q_mu,q_sqrt,pc,y_train,X,Z,mbs]=params
    cdrAtypes=cdr_types[0]
    cdrBtypes=cdr_types[1]
    d = int(X.shape[1]/sum(lmaxes))
    if len(Z)>0:
        is_sparse = True
    else:
        is_sparse = False

    [vas,vbs,cdr3as,cdr3bs,_,_] = read_vs_cdr3s_epis_subs(filename,va,vb,cdr3a,cdr3b,None,None,delimiter=delimiter)
    nseqs = max_len([vas,vbs,cdr3as,cdr3bs])
    
    Itest = np.ones((nseqs,),dtype=bool)
    seq_lists_letter = []
    seq_lists = []
    if 'cdr3' in cdrAtypes:
        cdr3as_letter,cdr3as, I = check_align_cdr3s(cdr3as,lmaxes[0])
        Itest = np.logical_and(Itest,I)
        seq_lists_letter.append(cdr3as_letter)
        seq_lists.append(tcrs2nums(cdr3as))
        
    if any([ c in ['cdr1','cdr2','cdr25'] for c in cdrAtypes]):
        cdrs = extract_cdrs_minimal_v(vas, organism, 'A', cdrAtypes, correct=True)
        for clist in cdrs:
            if len(clist)>0:
                seq_lists_letter.append(clist)
                cs, I = remove_starred_(clist)
                Itest = np.logical_and(Itest, I)
                seq_lists.append(tcrs2nums(cs))
        
    if 'cdr3' in cdrBtypes:
        cdr3bs_letter, cdr3bs, I = check_align_cdr3s(cdr3bs,lmaxes[len(cdrAtypes)])
        Itest = np.logical_and(Itest, I)
        seq_lists_letter.append(cdr3bs_letter)
        seq_lists.append(tcrs2nums(cdr3bs))
        
    if any([ c in ['cdr1','cdr2','cdr25'] for c in cdrBtypes]):
        cdrs = extract_cdrs_minimal_v(vbs, organism, 'B', cdrBtypes, correct=True)
        for clist in cdrs:
            if len(clist)>0:
                seq_lists_letter.append(clist)
                cs, I = remove_starred(clist)
                Itest = np.logical_and(Itest, I)
                seq_lists.append(tcrs2nums(cs))
          
    ncdrs=len(seq_lists)
    seq_lists_test = [[] for i in range(ncdrs)] 
    for ind in np.where(Itest)[0]:
        for i in range(ncdrs):
            seq_lists_test[i].append(seq_lists[i][ind])
     
    X_test = encode_with_pc(seq_lists_test,lmaxes,pc) 
    predictions = np.nan*np.ones((len(Itest),1))
    
    # Do predictions for valid sequences
    with tf.Session (graph=tf.Graph()):
        kernel = construct_rbf_kernel(d,lmaxes,lengthscales,variances)
        if is_sparse:
            m = gpflow.models.SVGP(X,y_train,kernel,gpflow.likelihoods.Bernoulli(),Z=Z,minibatch_size=mbs)
        else:
            m = gpflow.models.VGP(X,y_train,kernel,gpflow.likelihoods.Bernoulli())
        m.likelihood.variance = 1.0
        m.q_mu = q_mu
        m.q_sqrt = q_sqrt
        m.compile()
           
        p, _ = m.predict_y(X_test)
        predictions[Itest] = p
        
    return seq_lists_letter, predictions
