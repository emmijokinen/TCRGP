import numpy as np
import tensorflow as tf
import gpflow
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.metrics import roc_curve as roc_curve
from sklearn.decomposition import PCA as PCA
from scipy.cluster.vq import kmeans2
import csv
import matplotlib.pyplot as plt

# Use tf version 1.8. If using tf version >= 2, change its behaviour to tf 1 behaviour.
from packaging import version
if version.parse(tf.__version__) >= version.parse('2.0'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
else:
    import tensorflow as tf

alphabet='ARNDCEQGHILKMFPSTWYV-'
# subsmatfromAA2 assumes the alphabet has length 21 and that the last character is for the gap.
# Consider this when changing the alphabet

# Handle sequence data and features 

def max_len(seq_list):
    """Returns the maximum sequence length within the given list of lists."""
    lmax=0
    for seq in seq_list:
        lmax=max( lmax, len(seq) )
    return lmax

def tcrs2nums(tcrs):
    """Converts a list of (TCR) amino acid sequences to numbers. Each letter is changed to its index in the alphabet"""
    tcrs_num=[]
    n=len(tcrs)
    for i in range(n):
        t=tcrs[i]
        nums=[]
        for j in range(len(t)):
            nums.append(alphabet.index(t[j]))
        tcrs_num.append(nums)
    return tcrs_num

def nums2tcrs(nums):
    """Converts a list containing lists of numbers to amino acid sequences. Each number is considered to be an index of the alphabet."""
    tcrs_letter=[]
    n=len(nums)
    for i in range(n):
        num=nums[i]
        tcr=''
        for j in range(len(num)):
            tcr+=alphabet[num[j]]
        tcrs_letter.append(tcr)
    return tcrs_letter

def remove_starred(cdrs):
    """Returned cdrs are like the given cdrs, but cdrs with stars (*) are replaced by an empty entry.
    Ikeep contains the locations of cdrs which did not contain stars."""
    Ikeep = np.ones((len(cdrs),),dtype=bool)
    for i in range(len(cdrs)):
        if '*' in cdrs[i]:
            cdrs[i]=[]
            Ikeep[i]=False
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
        assert (l_max >= max_len(tcrs)), "Given max length must be greater than or equal to the max lenght of given sequences, "+str(max_len(tcrs))
    
    tcrs_aligned=[]
    for tcr in tcrs:
        tcrs_aligned.append(add_gap(tcr,l_max,gap_char))
    return tcrs_aligned

def check_align_cdr3s(cdr3s,lmaxtrain):
    """Check cdr3s for too long sequences or sequences containing characters outside alphabet
    returns cdr3s_letter (proper cdr3s aligned, but improper sequences are left as they are)
            cdr3s_aligned (proper cdr3s aligned, places of improper sequences are left empty),
            and Ikeep3 (locations of proper cdr3s)
    Here improper means sequences that are longer than those in the training data or contain
    characters outside the used alphabet."""
    lmaxtest=max_len(cdr3s) 
    Ikeep3=np.ones((len(cdr3s),),dtype=bool)
    cdr3s_aligned=[]
    cdr3s_letter =[]
    if lmaxtest>lmaxtrain:
        print('Maximum length of the given CDR3s is '+str(lmaxtest)+', but the maximum length is set to '+str(lmaxtrain)+'.')
        print('Longer sequences will be ignored.')   
        
    for i in range(len(cdr3s)):
        if len(cdr3s[i])<3 or len(cdr3s[i])>lmaxtrain or not all([ c in alphabet for c in cdr3s[i]]):
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
    Clipping should be done after the sequences are aligned."""
    for i in range(len(cdr3s)):
        cdr3s[i]=cdr3s[i][clip[0]:-clip[1]]
    return cdr3s

def subsmatFromAA2(identifier,data_file='data/aaindex2.txt'):
    """Retrieve a substitution matrix from AAindex2-file, scale it between 0 and 1, and add gap"""
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
    """Get first d pca-components from the given substitution matrix."""
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
    
def create_cdr_dict(alignment='imgt',species=['human'], alphabet_db_file_path='data/alphabeta_db.tsv'):
    """Creates a dictionary of the CDRs (1, 2, and 2.5) corresponding to each V-gene. 
    If alignment='imgt', the CDRs will be aligned according to imgt definitions.
    Dictionary has form cdrs12[organism][chain][V-gene] = [cdr1,cdr2,cdr2.5].
    alphabet_db_file_path: path of 'alphabeta_db.tsv' file, originally found in the TCRGP/data folder.
    """
    cdrs_all=file2dict(alphabet_db_file_path, key_fields=['organism','chain','region','id'],store_fields=['cdrs'])
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
    handles a few different formatsformats."""
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


def create_minimal_v_cdr_list(organism='human',chain='B',cdrtypes=['cdr1','cdr2','cdr25'],
                              alphabet_db_file_path='data/alphabeta_db.tsv'):
    """Create a list that determines minimal level of information (subgroup, name, allele) 
    needed to determine the wanted cdrtypes (cdr1, cdr2, cdr25).
    Possible organism are human and mouse and possible chains are A and B.
    alphabet_db_file_path: path of 'alphabeta_db.tsv' file, originally found in the TCRGP/data folder.
    """
    cdrs = create_cdr_dict(species=[organism], alphabet_db_file_path=alphabet_db_file_path)
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
        cs = c_array[Is,:][:,i_cs] # all CDRs with given subgroup
        c = ''.join(cs[0])
        if np.all(list(''.join(x)==c for x in cs)):
            vc_list.append([sub,'any','any',cs[0]])
        else:
            name_list=v_array[Is,2]
            names = np.unique(name_list)
            for name in names:
                In = name==name_list
                cn = cs[In] # all CDRs with given gene name
                c = ''.join(cn[0])
                if np.all(list(''.join(x)==c for x in cn)):
                    vc_list.append([sub,name,'any',cn[0]])
                else:        
                    alleles = v_array[Is,3][In]
                    for allele,c in zip(alleles,cn):
                        vc_list.append([sub,name,allele,c])
    return vc_list

def extract_cdrs_from_v(vgenes, organism, chain, cdrtypes,correctVs=True,check_v='none',
                        alphabet_db_file_path='data/alphabeta_db.tsv'):
    """Get requested cdrs (cdr_types) from the vgenes where possible. 
    If all requested cdrs could not be obtained from a vgenes, empty entries are returned in their place
    organism: human/mouse
    chain: A/B
    cdrtypes: some subset of ['cdr1','cdr2','cdr25']
    correctVs: If True, attempt to chage V-genes in correct format
    check_v: If 'none' accept only complete V-genes, if 'ignore', ignore incomplete V-genes (return empty CDRs),
             if 'deduce', try to deduce CDRs from incomplete V-genes, ignore where this fails.
    alphabet_db_file_path: path of 'alphabeta_db.tsv' file, originally found in the TCRGP/data folder.

    Returns lists of CDRs ([CDR1s, CDR2s, CDR2.5s]) determined from the V-genes. If CDRs for a V-gene 
        could not be obtained, the corresponding location is left empty.
    """
    if correctVs:
        vgenes=[correct_vgene(v,chain) for v in vgenes]
    
    cdrs12 = create_cdr_dict(species=[organism], alphabet_db_file_path=alphabet_db_file_path)
    vc_list = create_minimal_v_cdr_list(organism,chain,cdrtypes, alphabet_db_file_path=alphabet_db_file_path)
    
    ics = []
    if 'cdr1' in cdrtypes:
        ics.append(0)
    if 'cdr2' in cdrtypes:
        ics.append(1)
    if 'cdr25' in cdrtypes:
        ics.append(2)
    
    cdrs=[[],[],[]] 
    for v in vgenes:
        try:
            cs = cdrs12[organism][chain][v]
            for i in ics:
                cdrs[i].append(cs[i])
        except KeyError:
            if check_v=='none':
                raise KeyError("Invalid V-gene: "+v+". Use check_v='ignore' to ignore TCRs with invalid V-genes, or check_v='deduce' to try deducing requsted CDRs.")
            notFound=True
            if check_v=='deduce': 
                _,sub,name,allele=split_v(v)
                
                for row in vc_list:
                    if ( (row[0]==sub) and ( (row[1]==name) or (row[1]=='any') or (name=='1' and row[1]=='') )
                        and (row[2]==allele or row[2]=='any') ):

                        for i in ics:
                            cdrs[i].append(row[3][i])
                        notFound=False
                        continue
            if notFound:
                for i in ics:
                    cdrs[i].append([])   
    return cdrs

def read_vs_cdr3s_epis_subs(datafile,va='va',vb='vb',cdr3a='cdr3a',cdr3b='cdr3b', epis='epitope',subs='subject',delimiter=',',encoding='bytes'):
    """Reads VA-genes, VB-genes, CDR3As, CDR3Bs from the given data file. 
    The columns are determined by va, vb, cdr3a, and cdr3b. Any of them can also be None, 
    if they are not required.
    
    Returns a list of lists [vas, vbs, cdr3as, cdr3bs, epis, subs]. If a specifier was None, 
    the corresponding list is empty."""
    
    with open(datafile, newline='') as csvfile:
        header = csvfile.readline()
    fields=header.strip().split(delimiter)
    
    cols=[]
    names = [va,vb,cdr3a,cdr3b,epis,subs]
    i_names = []
    for i, name in enumerate(names):
        if name is not None:
            try:
                cols.append(fields.index(name))
                i_names.append(i)
            except ValueError:
                print('Check your header names (va, vb, cdr3a, cdr3b), \''+name+'\' was not found from the datafile.')
                raise
        
    va_vb_3a_3b_ep_su =  [[] for i in range(6)]
    lists = np.loadtxt(datafile,dtype=str,delimiter=delimiter,comments=None,unpack=True,skiprows=1,usecols=cols,encoding=encoding)
    if len(cols)==1:
        va_vb_3a_3b_ep_su[i_names[0]]=lists
    else:
        for i_names, li in zip(i_names,lists):
            va_vb_3a_3b_ep_su[i_names] = li
    
    return va_vb_3a_3b_ep_su

def get_sequence_lists(datafile,organism,epi,cdr_types,delimiter,clip,lmax3=None,
                       va='va',vb='vb', cdr3a='cdr3a',cdr3b='cdr3b',epis='epitope',subs='subject',
                       check_v='none',balance_controls=True,encoding='bytes',
                       alphabet_db_file_path='data/alphabeta_db.tsv'):
    """Get sequence lists of the requested cdr_types from datafile
    organism: human/mouse
    epi: epitope name in datafile (ignored if balance_controls=False)
    cdrtypes: some subset of ['cdr1','cdr2','cdr25']
    delimiter: delimiter used in datafile, e.g. ','
    clip: [a,b] remove a AAs from beginning and b AAs from the end of the given cdr3s
    lmax3: determines maximum length for CDR3s. Can be given separately for cdr3A and cdr3B. If None, maximum 
           length of CDR3s in the data file is used.
    va,vb,cdr3a,cdr3b,epis,subs: column names of VA-genes, VB-genes, CDR3A, CDR3B, epitopes, subjects. Can be None, if not required.
    check_v: If 'none' accept only complete V-genes, if 'ignore', ignore incomplete V-genes (return empty CDRs),
             if 'deduce', try to deduce CDRs from incomplete V-genes, ignore where this fails.
    balance_controls: if True, when epitope-specific TCRs are removed, remove also correponding amount of control TCRs.
    alphabet_db_file_path: path of 'alphabeta_db.tsv' file, originally found in the TCRGP/data folder.

    Returns epitopes, subjects, cdr_lists, lmaxes, Itest (which indicates which of the given TCRs were returned)
    """
    # Read data file and extract requested information
    vas,vbs,cdr3as,cdr3bs,epitopes,subjects = read_vs_cdr3s_epis_subs(datafile,va=va,vb=vb,cdr3a=cdr3a,cdr3b=cdr3b, epis=epis,subs=subs,delimiter=delimiter,encoding=encoding)
    if balance_controls:
        Ie = epitopes == epi
    if isinstance(lmax3,int):
        lmax3=[lmax3,lmax3]
    elif lmax3==None :
        lmax3=[max_len(cdr3as),max_len(cdr3bs)]
    elif lmax3[0]==None: 
        lmax3[0]=max_len(cdr3as)
    elif lmax3[1]==None: 
        lmax3[1]=max_len(cdr3bs)
    clip3=sum(clip)>0
    # Get and check CDRs from epitope-specific TCRs
    Itest = np.ones((max_len([epitopes,cdr3as,cdr3bs])),dtype=bool)
    seq_lists = []
    if 'cdr3' in cdr_types[0]:
        cdr3as_letter,cdr3as, I = check_align_cdr3s(cdr3as,lmax3[0]) # I: which cdr3s will be kept
        if balance_controls:
            Itest[np.logical_and( Ie,~I)]=False
            Itest[~Ie] = Itest[Ie]
        else:
            Itest[~I]=False
        if clip3:
            cdr3as = clip_cdr3s(cdr3as,clip)
        seq_lists.append(tcrs2nums(cdr3as))
        
    if any([ c in ['cdr1','cdr2','cdr25'] for c in cdr_types[0]]):
        cdrs = extract_cdrs_from_v(vas,organism,'A',cdr_types[0],correctVs=True,check_v=check_v, alphabet_db_file_path=alphabet_db_file_path)
        for clist in cdrs:
            if len(clist)>0:
                cs, I = remove_starred(clist)
                seq_lists.append(tcrs2nums(cs))
                I=np.asarray([len(x)>0 for x in cs])
                if balance_controls:
                    Itest[np.logical_and( Ie,~I)]=False
                    Itest[~Ie] = Itest[Ie]
                else:
                    Itest[~I]=False
        
    if 'cdr3' in cdr_types[1]:
        _, cdr3bs, I = check_align_cdr3s(cdr3bs,lmax3[1])
        if balance_controls:
            Itest[np.logical_and( Ie,~I)]=False
            Itest[~Ie] = Itest[Ie]
        else:
            Itest[~I]=False
        if clip3:
            cdr3bs = clip_cdr3s(cdr3bs,clip)
        seq_lists.append(tcrs2nums(cdr3bs))
        
    if any([ c in ['cdr1','cdr2','cdr25'] for c in cdr_types[1]]):
        cdrs = extract_cdrs_from_v(vbs,organism,'B',cdr_types[1],correctVs=True,check_v=check_v, alphabet_db_file_path=alphabet_db_file_path)
        for clist in cdrs:
            if len(clist)>0:
                cs, I = remove_starred(clist)
                seq_lists.append(tcrs2nums(cs))
                I=np.asarray([len(x)>0 for x in cs])
                if balance_controls:
                    Itest[np.logical_and( Ie,~I)]=False
                    Itest[~Ie] = Itest[Ie]
                else:
                    Itest[~I]=False

    assert (sum(Itest) >0), "Given data didn't contain any TCRs with the required information. You may try to use check_v='deduce' or train a model with only CDR3s."
    ncdrs=len(seq_lists)
    cdr_lists = [[] for i in range(ncdrs)]    
    for ind in np.where(Itest)[0]:
        for i in range(ncdrs):
            cdr_lists[i].append(seq_lists[i][ind])
    
    lmaxes=[]
    for i in range(ncdrs):
        lmaxes.append(len(cdr_lists[i][0]))
    
    if len(subjects)>0:
        subjects=np.asarray(subjects)[Itest]
    if len(epitopes)>0:
        epitopes=np.asarray(epitopes)[Itest]
    
    return epitopes, subjects, cdr_lists, lmaxes, Itest

def get_subjects(organism,epi,epitopes,subjects,min_subjects=5, cv=5):
    """Get subject list for given epitope. Define new subjects if there are not enough.
    This function is primarily inteded to be used from within loso"""
    I = epitopes==epi
    subjects_epi = subjects[I]
    subjects_u = np.unique(subjects_epi)  
    l_epis = sum(I)
    n_subs = len(subjects_u)
        
    print(organism+' '+epi+': '+str(n_subs)+ ' subjects, '+str(l_epis)+' positive and '+str(sum(~I))+' control samples')

    ind1 = 0
    if n_subs < min_subjects:
        print('Not enough subjects. Using ' +str(cv)+ '-fold cross-validation')
        subjects_u=list(range(1,cv+1))
        subjects_epi = np.asarray((subjects_u*int(np.ceil(l_epis/cv)))[:l_epis])    
        n_subs=cv

    return subjects_epi, l_epis, subjects_u, n_subs, I

def get_stratified_folds(y,k=200):
    """
    y: numpy array of size (n,1), consists of zeros and ones (ones for positive samples)
    k: number of folds
    Returns inds: indices for k folds.
    """
    y=y.copy()
    npos = np.sum(y)
    nneg = len(y)-npos
    inds = np.zeros((len(y),),dtype=int)
    for i in range(k):
        n_p = int(npos/(k-i))
        n_n = int(nneg/(k-i))
 
        ii = np.where(y==1)[0]
        I=np.random.choice(ii,n_p,replace=False)
        inds[I]=i
        y[I]=2
        
        ii = np.where(y==0)[0]
        I = np.random.choice(ii,n_n,replace=False)
        inds[I]=i
        y[I]=2
        
        npos -= n_p
        nneg -= n_n
        
    return inds 


# Plotting

def plot_aurocs_ths(y_list,p_list,epi='',thresholds=[0.0, 0.05, 0.1, 0.2],dpi=200,figsize=(10,3),
                    save_plot_path=None, return_best_threshold=False):
    """plot AUROC and threshold values
    y_list: list of arrays or array of class labels (1,0). If list of arrays, different ROC is plotted for each array.
    p_list: list of arrays or array of predictions.
    epi: name of the epitope or other string to be added in the beginning of the figure title.
    thresholds: False positive rates for which prediction thresholds will be shown.
    save_plot_path: Path for saving the final plot. If None, the plot will be shown.
    return_best_threshold: If True, adds to the returned variables list the value of the best threshold found.

    Returns mean AUROC, mean weighted AUROC and AUROC across all folds, in addition to producing the figure.
    If return_best_threshold is True, also returns the best threshold.
    """
    
    f=plt.figure(figsize=figsize,dpi=dpi)
    if type(y_list) is list:
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
    else:
        y_all = y_list
        p_all = p_list
        mean_auc=None
        mean_wt_auc=None
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

    auc_all=roc_auc(y_all,p_all)
    plt.title(epi+' AUROC: {:1.4f}'.format(auc_all))
    plt.legend(handles=legs,labels=labels,loc=(1.1,0.6))   

    if save_plot_path is not None:
        plt.savefig(save_plot_path, dpi=500, bbox_inches='tight')
    else:
        plt.show()

    if return_best_threshold==False:
        return mean_auc, mean_wt_auc, auc_all
    else:
        return mean_auc, mean_wt_auc, auc_all, th[i_best]

# Construct kernels, select inducing points, train and load models

def construct_rbf_kernel(d,lmaxes,lengthscales=[1.0],kernvaris=[1.0]):
    """Contsructs a kernel as as sum of GPFlow RBF-kernels with given lengthscales and variances.
    d: number of features used
    lmaxes: list of the lengths of the CDRs used.
    lengthscales: list of (initial) lengthscales for RBF-kernels for each CDR in the following order: cdr3a, cdr1a, cdr2a, cdr25a, 
        cdr3b, cdr1b, cdr2b, cdr25b. If len(lengthscales)==1, but many CDRs are used, the same lengthscale 
        is used for each of them 
    kernvaris: list of (initial) kernel variances. Same format as with lengthscales 
    
    Returns the constructed kernel
    """
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
    """Select inducing point locations with kmeans from training data XP_tr, and minibatch size mbs.
    n_tr = number of training points.
    If nZ<1, there will be nZ * n_tr inducing points. Otherwise there will be nZ training points. 
    Same applies for the minibatch size mbs, except that if mbs=0 or mbs > n_tr, mbs is set to n_tr
    
    Returns inducing points and minibatch size
    """  
    n_tr = XP_tr.shape[0]
    if nZ < 1:
        nZ = int(np.ceil(nZ*n_tr))
    Z = kmeans2(XP_tr, nZ, minit='points')[0]
    if mbs == 0 or mbs > n_tr:
        mbs = n_tr # use all data
    elif mbs < 1:
        mbs = int(np.ceil(mbs*n_tr))
    return Z, mbs

def print_model_info(model):
    """Print information of the parameters of a model created with train_classifier-function"""
    [cdr_types,lmaxes,lengthscales,variances,_,_,_,y,_,Z,mbs,clip]=model
    print('CDR types: ',end='')
    cdr_list=[]
    for i,chain in enumerate(['a','b']):
        for cdr in ['cdr3','cdr1','cdr2','cdr25']:
            if cdr in cdr_types[i]:
                print(cdr+chain, end=' ')
                cdr_list.append(cdr+chain)
    print('\nmax lengths: ',end='')
    if 'cdr3' in cdr_types[0]:
        print('cdr3a: {:d}'.format(model[1][0]), end=' ')
    if 'cdr3' in cdr_types[1]:
        print('cdr3b: {:d}'.format(model[1][len(cdr_types[0])]))
    print('number of training samples: {:d} ({:d} positive, {:d} negative)'.format(len(y),np.sum(y),len(y)-np.sum(y)))
    print('number of inducing points: {:d}'.format(len(model[9])))
    if len(model[10])==0:
        print('minibatch size: 0')
    else:
        print('minibatch size: {:d}'.format(model[10]))
    print('lengthscales: ',end='')
    for i,c in enumerate(cdr_list):
        print('{:s}: {:.4f}'.format(c,lengthscales[i]),end=' ')
    print('\nkernel variances: ',end='')
    
    for i,c in enumerate(cdr_list):
        print('{:s}: {:.4f}'.format(c,variances[i]),end=' ')
    print('\nclip cdr3s: {:d} from beginning, {:d} from end'.format(clip[0],clip[1]))
                          
def loso(datafile,organism,epi,pc,cdr_types=[[],['cdr3']],l=[1.0],var=[1.0],
         m_iters=5000,lr=0.005,nZ=0,mbs=0, clip=[0,0],min_subjects=5,cv=5,delim=',',
         va='va',vb='vb',cdr3a='cdr3a',cdr3b='cdr3b', subs='subject',epis='epitope', check_v='none',
         save_plots_path=None, return_best_threshold=False, balance_controls=True,
         alphabet_db_file_path='data/alphabeta_db.tsv'):
    """
    Leave-on-subject-out cross-validation with TCRGP
    datafile: delimeted file which contains columns Epitope, Subject, va, vb, cdr3a, cdr3b. If some of them are not
        required to get the requsted cdr types, they may be empty.
    organism: 'human' or 'mouse' 
    epi: name of the epitope
    pc: principal components or features for each amino acid.
    cdr_types: CDRs utilized by the model. list that contains list of CDR types for chain A and chain B. 
        possible CDR types are cdr1, cdr2, cdr25 and cdr3.    
    l: list of initial lengthscales for RBF-kernels for each CDR in the following order: cdr3a, cdr1a, cdr2a, cdr25a, 
        cdr3b, cdr1b, cdr2b, cdr25b. If len(l)==1, but many CDRs are used, the same lengthscale 
        is used for each of them 
    var: list of initial kernel variances (weights). Same format as with l.
    m_iters: maximum number of iterations
    lr: learning rate for Adam optimizer
    nZ: number of inducing points to be used with SVGP(selected with kmeans). If zero, VGP will be used.
    mbs: minibatch size, in case SVGP is used.
    clip: list, remove clip[0] amino acids from the beginnings and clip[1] amino acids from the ends of CDR3s
    min_subjects: minimum number of subjects required for loso-cv. If there are less subjects, 
        do cv-fold cross-validation instead.
    cv: how many fold cross-cross validation in case loso is not possible.
    delim: delimiter used in datafile
    va,vb,cdr3a,cdr3b,sub,epis: names for the columns that contain information for VA-genes, VB-genes, CDR3As, CDR3Bs,
        subjects, and epitopes. Any of them can be None, if they are not required to get the requested cdr_types 
    check_v: if 'none', no checking is done, if the v-gene is incomplete (e.g. no allele is given), the function will fail. 
            If 'ignore', TCRs with incomplete V-genes are ignored. If 'deduce' all TCRs with V-genes from which the 
            requested CDRs (CDR1, CDR2, CDR2.5) can be deduced from, are utilized, and other TCRs are ignored.
    save_plots_path: str. path for saving the ROC plot.
    return_best_threshold: boolean. If true, function returns also the best threshold that was found from the ROC plot.
    balance_controls: if True, when epitope-specific TCRs are removed, remove also corresponding amount of control TCRs.
    alphabet_db_file_path: path of 'alphabeta_db.tsv' file, originally found in the TCRGP/data folder.

    Returns mean AUC, mean weighted AUC, class lists for all subjects/folds, predictions for all subjects/folds and plots the AUROCs, (best threshold, only if return_best_threshold=True)
    """
    
    # Read data file and extract requested CDRs
    epitopes,subjects,cdr_lists,lmaxes,_ = get_sequence_lists(datafile,organism,epi,cdr_types,delim,clip,None,va,vb,
                                                              cdr3a,cdr3b,epis,subs,check_v=check_v,
                                                              balance_controls=balance_controls,
                                                              alphabet_db_file_path=alphabet_db_file_path)

    # encode with pc components
    d = pc.shape[0]
    X = encode_with_pc(cdr_lists,lmaxes,pc)
    
    # Handle subject list that determins the training folds.
    subjects_epi, l_epi, subjects_u, n_subs, Ipos = get_subjects(organism,epi,epitopes,subjects,min_subjects,cv)
    
    Ineg = ~Ipos
    y = np.zeros((len(epitopes),1), dtype=int)
    y[Ipos] = 1
      
    y_list, p_list = [], []
    i=1
    for subject in subjects_u:
        
        Isub = np.ones((l_epi),dtype=bool)
        Isub[subjects_epi==subject] = False
        
        I = np.ones((2*l_epi),dtype=bool)
        I[Ipos]= Isub
        I[Ineg]= Isub
        
        with tf.Session(graph=tf.Graph()):
            kernel = construct_rbf_kernel(d,lmaxes,l,var)
            if nZ == 0: # use VGP
                m = gpflow.models.VGP(X[I,:],y[I],kernel,gpflow.likelihoods.Bernoulli())
            else: # use SVGP
                # inducing locations by kmeans
                Z, mbs = select_Z_mbs(nZ,mbs,X[I,:])
                m = gpflow.models.SVGP(X[I,:],y[I],kernel,gpflow.likelihoods.Bernoulli(),Z=Z,minibatch_size=mbs)
            m.likelihood.variance = 1.0

            print('\rComputing fold: {:d}/{:d}'.format(i,n_subs),end='')
            i+=1
            gpflow.train.AdamOptimizer(lr).minimize(m, maxiter=m_iters)
            p, _ = m.predict_y(X[~I,:])

        y_list.append(y[~I])
        p_list.append(p)
    print('\rAll folds ({:d}) computed.  '.format(n_subs))

    if return_best_threshold==False:
        mean_auc, mean_wt_auc, auc_all = plot_aurocs_ths(y_list,p_list,epi, save_plot_path=save_plots_path,
                                                         return_best_threshold=return_best_threshold)
        return [mean_auc, mean_wt_auc, y_list, p_list]
    else:
        mean_auc, mean_wt_auc, auc_all, best_threshold = plot_aurocs_ths(y_list,p_list,epi, save_plot_path=save_plots_path,
                                                                         return_best_threshold=return_best_threshold)
        return [mean_auc, mean_wt_auc, y_list, p_list, best_threshold]


def loo(datafile,organism,epi,pc,cdr_types=[[],['cdr3']],l=[1.0],var=[1.0],m_iters=5000,lr=0.005,nZ=0,mbs=0,
        clip=[0,0],delim=',', va='va',vb='vb',cdr3a='cdr3a',cdr3b='cdr3b', subs=None,epis='epitope',
        check_v='none',balance_controls=True, alphabet_db_file_path='data/alphabeta_db.tsv'):
    """
    Leave-on-out cross-validation with TCRGP
    datafile: delimeted file which contains columns Epitope, Subject, va, vb, cdr3a, cdr3b. If some of them are not
        required to get the requsted cdr types, they may be empty.
    organism: 'human' or 'mouse' 
    epi: name of the epitope
    pc: principal components or features for each amino acid.
    cdr_types: CDRs utilized by the model. list that contains list of CDR types for chain A and chain B. 
        possible CDR types are cdr1, cdr2, cdr25 and cdr3.
    l: list of initial lengthscales for RBF-kernels for each CDR in the following order: cdr3a, cdr1a, cdr2a, cdr25a, 
        cdr3b, cdr1b, cdr2b, cdr25b. If len(l)==1, but many CDRs are used, the same lengthscale 
        is used for each of them 
    var: list of initial kernel variances (weights). Same format as with l.
    m_iters: maximum number of iterations
    lr: learning rate for Adam optimizer
    nZ: number of inducing points to be used with SVGP(selected with kmeans). If zero, VGP will be used.
    mbs: minibatch size, in case SVGP is used.
    clip: list, remove clip[0] amino acids from beginning and clip[1] amino acids from the end
    delim: delimiter used in datafile
    va,vb,cdr3a,cdr3b,subs,epis: names for the columns that contain information for VA-genes, VB-genes, CDR3As, CDR3Bs,
        subjects, and epitopes. Any of them can be None, if they are not required to get the requested cdr_types 
    check_v: If 'none' accept only complete V-genes, if 'ignore', ignore incomplete V-genes (return empty CDRs),
             if 'deduce', try to deduce CDRs from incomplete V-genes, ignore where this fails.
    balance_controls: if True, when epitope-specific TCRs are removed, remove also correponding amount of control TCRs.
    alphabet_db_file_path: path of 'alphabeta_db.tsv' file, originally found in the TCRGP/data folder.

    Returns AUROC, list of classes, and list of predictions
    """
    
    # Read data file and extract requested CDRs
    epitopes,subjects,cdr_lists,lmaxes,_ = get_sequence_lists(datafile,organism,epi,cdr_types,delim,clip,None, va,vb,cdr3a,cdr3b,epis,subs,check_v=check_v,balance_controls=balance_controls, alphabet_db_file_path=alphabet_db_file_path)

    # encode with pc components
    d = pc.shape[0]
    X = encode_with_pc(cdr_lists,lmaxes,pc)
    n = len(epitopes)
    
    Ipos = epitopes==epi
    l_epi = sum(Ipos)
 
    print(str(l_epi)+' positive samples')
    Ineg = ~Ipos
    y = np.zeros((n,1), dtype=int)
    y[Ipos] = 1
      
    ps = np.zeros((n,1),dtype=float)
    i=1
    for ind in range(n):
        
        I = np.ones((n,),dtype=bool)
        I[ind]= False
        
        with tf.Session(graph=tf.Graph()):
            kernel = construct_rbf_kernel(d,lmaxes,l,var)
            if nZ == 0: # use VGP
                m = gpflow.models.VGP(X[I,:],y[I],kernel,gpflow.likelihoods.Bernoulli())
            else: # use SVGP
                # inducing locations by kmeans
                Z, mbs = select_Z_mbs(nZ,mbs,X[I,:])
                m = gpflow.models.SVGP(X[I,:],y[I],kernel,gpflow.likelihoods.Bernoulli(),Z=Z,minibatch_size=mbs)
            m.likelihood.variance = 1.0

            print('\rComputing fold: {:d}/{:d}'.format(i,n),end='')
            i+=1
            
            gpflow.train.AdamOptimizer(lr).minimize(m, maxiter=m_iters)
            p, _ = m.predict_y(X[~I,:])

        ps[ind]=p
    
    print('\rAll folds ({:d}) computed.  '.format(n))
    _, _, auc_all = plot_aurocs_ths(y,ps,epi,dpi=150)
    return [auc_all, y, ps]

def kfold_stratified(datafile,organism,epi,pc,k=200,cdr_types=[[],['cdr3']],l=[1.0],var=[1.0],m_iters=5000,lr=0.005,nZ=0,mbs=0,clip=[0,0],delim=',', va='va',vb='vb',cdr3a='cdr3a',cdr3b='cdr3b', subs=None,epis='epitope',check_v='none',balance_controls=False, balance_tr_controls=True):
    """
    Stratified k-fold cross-validation with TCRGP
    datafile: delimeted file which contains columns Epitope, Subject, va, vb, cdr3a, cdr3b. If some of them are not
        required to get the requsted cdr types, they may be empty.
    organism: 'human' or 'mouse' 
    epi: name of the epitope
    pc: principal components or features for each amino acid.
    k: number of folds
    cdr_types: CDRs utilized by the model. list that contains list of CDR types for chain A and chain B. 
        possible CDR types are cdr1, cdr2, cdr25 and cdr3.
    l: list of initial lengthscales for RBF-kernels for each CDR in the following order: cdr3a, cdr1a, cdr2a, cdr25a, 
        cdr3b, cdr1b, cdr2b, cdr25b. If len(l)==1, but many CDRs are used, the same lengthscale 
        is used for each of them 
    var: list of initial kernel variances (weights). Same format as with l.    
    m_iters: maximum number of iterations
    lr: learning rate for Adam optimizer
    nZ: number of inducing points to be used with SVGP(selected with kmeans). If zero, VGP will be used.
    mbs: minibatch size, in case SVGP is used.
    clip: list, remove clip[0] amino acids from beginning and clip[1] amino acids from the end
    delim: delimiter used in datafile
    va,vb,cdr3a,cdr3b,sub,epis: names for the columns that contain information for VA-genes, VB-genes, CDR3As, CDR3Bs,
        subjects, and epitopes. Any of them can be None, if they are not required to get the requested cdr_types 
    check_v: If 'none' accept only complete V-genes, if 'ignore', ignore incomplete V-genes (return empty CDRs),
             if 'deduce', try to deduce CDRs from incomplete V-genes, ignore where this fails.
    balance_controls: get same amount of positive and negative TCRs
    balance_tr_controls: use same amount of positive and negative TCRs (even if there are more negative TCRs for testing)
    
    Returns classes, fold indices, and predictions for the TCRs in the datafile.
    """
    
    # Read data file and extract requested CDRs
    epitopes,subjects,cdr_lists,lmaxes,_ = get_sequence_lists(datafile,organism,epi,cdr_types,delim,clip,None, va,vb,cdr3a,cdr3b,epis,subs,check_v=check_v,balance_controls=balance_controls)

    # encode with pc components
    d = pc.shape[0]
    X = encode_with_pc(cdr_lists,lmaxes,pc)
    n = len(epitopes)
    
    Ipos = epitopes==epi
    l_epi = sum(Ipos)
 
    print(str(l_epi)+' positive and '+str(sum(~Ipos))+' negative samples')
    inds_pos=np.nonzero(Ipos)[0]
    Ineg = ~Ipos
    y = np.zeros((n,1), dtype=int)
    y[Ipos] = 1
    
    inds= get_stratified_folds(y,k=k)
    ps = np.ones((n,1),dtype=float)*np.nan

    for ind in range(k):
        
        Itest = np.zeros((n,),dtype=bool)
        Itest[inds==ind]= True
        Itrain = ~Itest
        if balance_tr_controls:
            # only use subset of negative (number of epitope-specific TCRs)
            Itrain[np.random.choice(np.where(Itrain & ~Ipos)[0],sum(Itrain[~Ipos])-sum(Itrain[Ipos]),replace=False)] = False
        
        with tf.Session(graph=tf.Graph()):
            kernel = construct_rbf_kernel(d,lmaxes,l,var)
            if nZ == 0: # use VGP
                m = gpflow.models.VGP(X[Itrain,:],y[Itrain],kernel,gpflow.likelihoods.Bernoulli())
            else: # use SVGP
                # inducing locations by kmeans
                Z, mbs = select_Z_mbs(nZ,mbs,X[Itrain,:])
                m = gpflow.models.SVGP(X[Itrain,:],y[Itrain],kernel,gpflow.likelihoods.Bernoulli(),Z=Z,minibatch_size=mbs)
            m.likelihood.variance = 1.0
            
            print('\rComputing fold: {:d}/{:d}'.format(ind+1,k),end='')
            
            gpflow.train.AdamOptimizer(lr).minimize(m, maxiter=m_iters)
            p, _ = m.predict_y(X[Itest,:])

        ps[Itest]=p
        
    print('\rAll folds ({:d}) computed.  '.format(k))
    _, _, auc_all = plot_aurocs_ths(y,ps,epi,dpi=150)
    
    return y, inds, ps


def train_classifier(datafile,organism,epi,pc, cdr_types=[[],['cdr3']],
                     l=[1.0],var=[1.0],m_iters=5000,lr=0.005,nZ=0,mbs=0,lmax3=None,
                     clip=[0,0],delimiter=',', va='va',vb='vb',cdr3a='cdr3a',cdr3b='cdr3b',epis='epitope',
                     check_v='none',balance_controls=True, return_preds=False,
                     alphabet_db_file_path='data/alphabeta_db.tsv'):
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
    lr: learning rate for Adam optimizer
    nZ: number of inducing points to be used with SVGP(selected with kmeans). If zero, VGP will be used.
    mbs: minibatch size, in case SVGP is used.
    lmax3: maximum length for CDR3s. If None, this is determined by the longest CDR3 in the datafile.
    clip: list, remove clip[0] amino acids from beginning and clip[1] amino acids from the end
    delimiter: delimiter used in datafile
    va,vb,cdr3a,cdr3b,epis: names for the columns that contain information for VA-genes, VB-genes, CDR3As, CDR3Bs,
        and epitopes. Any of them can be None, if they are not required to get the requested cdr_types
    check_v: If 'none' accept only complete V-genes, if 'ignore', ignore incomplete V-genes (return empty CDRs),
             if 'deduce', try to deduce CDRs from incomplete V-genes, ignore where this fails.
    balance_controls: if True, when epitope-specific TCRs are removed, remove also correponding amount of control TCRs.
    return_preds: if True, adds the predicted values from the classifier to the returned values.
    alphabet_db_file_path: path of 'alphabeta_db.tsv' file, originally found in the TCRGP/data folder.

    Returns AUROC (for the training data) and parameters of the trained model.
    If return_preds=True, returns predictions as well.
    """
     
    # Read data file and extract requested CDRs
    epitopes,_,cdr_lists,lmaxes,_ = get_sequence_lists(datafile,organism,epi,cdr_types,
                                                       delimiter,clip,lmax3,
                                                       va,vb,cdr3a,cdr3b,epis,subs=None,
                                                       check_v=check_v,balance_controls=balance_controls,
                                                       alphabet_db_file_path=alphabet_db_file_path)
    
    # Class labels
    y = np.zeros((len(epitopes),1),dtype=int)
    y[epitopes==epi] = 1
    assert any(y),"Given epitope did not occur in the training data. There we no positive samples."
    
    # encode with pc components
    d = pc.shape[0]
    X = encode_with_pc(cdr_lists,lmaxes,pc)

    with tf.Session(graph=tf.Graph()):
        kernel = construct_rbf_kernel(d,lmaxes,l,var)
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
                try:
                    ls.append(m.kern.kernels[i].lengthscales.value)
                    vs.append(m.kern.kernels[i].variance.value)
                except AttributeError: # some tf versions have kern_list instead of kernels
                    ls.append(m.kern.kern_list[i].lengthscales.value)
                    vs.append(m.kern.kern_list[i].variance.value)
        else:
            ls.append(m.kern.lengthscales.value)
            vs.append(m.kern.variance.value)

        param_list = [cdr_types,lmaxes,ls, vs, m.q_mu.value, m.q_sqrt.value, pc, y, X]
        if nZ>0:
            param_list.append(m.feature.Z.value)
            param_list.append(mbs)
        else:
            param_list.append([])
            param_list.append([])
        param_list.append(clip)
        
        p, _ = m.predict_y(X)
    
    auc = roc_auc(y,p) # Training AUC

    if return_preds==False:
        return auc, param_list
    else:
        return auc, param_list, p

def predict(datafile, params, organism='human',va=None,vb=None,cdr3a=None,cdr3b='cdr3b',
            delimiter=',',encoding='bytes',check_v='none', alphabet_db_file_path='data/alphabeta_db.tsv'):
    """Do predictions for the TCRs in the given file. 
    Predictions are returned in the same order as the TCRs in the file.
    datafile: name of the file containing the TCRs for testing
    params: parameters needed to rebuild the classification model. 
        This is can be obtained from the train_classifier-function
    organism: human or mouse
    va,vb,cdr3a,cdr3b: names for the columns that contain information for VA-genes, VB-genes, CDR3As, and CDR3Bs.
        Any of them can be None, if they are not required to get the requested cdr_types
    delimiter: delimiter used in datafile
    encoding: encoding used in data file.
    check_v: If 'none' accept only complete V-genes, if 'ignore', ignore incomplete V-genes (return empty CDRs),
             if 'deduce', try to deduce CDRs from incomplete V-genes, ignore where this fails.
    alphabet_db_file_path: path of 'alphabeta_db.tsv' file, originally found in the TCRGP/data folder.

    Returns predictions in the same order as the TCRs appear in datafile (nan for TCRs for which no prediction could be made).
    """
    # Extract parameters
    [cdr_types,lmaxes,lengthscales,variances,q_mu,q_sqrt,pc,y_train,X,Z,mbs,clip]=params
    d = int(X.shape[1]/sum(lmaxes))
    if len(Z)>0:
        is_sparse = True
    else:
        is_sparse = False

    num_cdrs_alpha = 0
    lmax3_alpha = 0
    lmax3_beta = 0
    if 'cdr3' in cdr_types[0]:
        lmax3_alpha = lmaxes[cdr_types[0].index('cdr3')]
    if 'cdr3' in cdr_types[1]:
        lmax3_beta = lmaxes[num_cdrs_alpha + cdr_types[1].index('cdr3')]
    lmax3 = [lmax3_alpha, lmax3_beta]

    # Read data file and extract requested CDRs
    _,_,cdr_lists,lmaxes,Itest = get_sequence_lists(datafile,organism,None,cdr_types,delimiter,clip,lmax3,
                                                    va,vb,cdr3a,cdr3b,epis=None,subs=None,check_v=check_v,
                                                    balance_controls=False,encoding=encoding,
                                                    alphabet_db_file_path=alphabet_db_file_path)
                
    X_test = encode_with_pc(cdr_lists,lmaxes,pc) 
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
    
    # return sequence lists and predictions in original order
    return predictions

# Diversity computations

def diversity_K(K,mintcrs=2):
    """
    Diversity of TCRs whose covariance matrix K is.
    Nan is returned if there are less than mintcrs TCRs.
    """
    n = K.shape[0]
    if n < mintcrs:
        return np.nan
    div = ( sum(K[np.triu_indices(n,1)])/(0.5*(n-1)*n) )**(-1)
    return div

def diversity_K2(K,mintcrs=2):
    """
    Diversity between TCRs on (n) rows and (m) columns, whose covariance matrix K (size n*m) is. (e.g. between two subjects)
    Nan is returned if n or m is smaller than mintcrs.
    """
    n1,n2 = K.shape
    if n1 < mintcrs and n2 < mintcrs:
        return np.nan
    N=n1*n2 # number of comparisons
    ksum=np.sum(K)
    div = ( ksum/N )**(-1)
    return div

def diversity_Kmany(K,Isubs,mintcrs=2):
    """
    Diversity of TCRs between subjects (determined by Isubs), whose covariance matrix K is.
    Nan is returned if there are less than mintcrs TCRs.
    """
    n = K.shape[0]
    if n < mintcrs:
        return np.nan
    kvals=[]
    usubs=np.unique(Isubs)
    N=0 # number of comparisons
    ksum=0
    for i in range(len(usubs)-1):
        for j in range(i,len(usubs)):
            Ii=Isubs==usubs[i]
            Ij=Isubs==usubs[j]
            ksum+=np.sum(K[Ii,:][:,Ij])
            N+=sum(Ii)*sum(Ij)
    
    div = ( ksum/N )**(-1)
    return div
