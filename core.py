def parse_map(b):
    rows = []
    text = _rt(b)
    for line in text.splitlines():
        line = line.strip()
        # 1. Ignorer les lignes vides et les commentaires (#)
        if not line or line.startswith("#"):
            continue
        
        # 2. Gérer les virgules, tabulations ou espaces
        if "," in line and "\t" not in line:
            p = [x.strip() for x in line.split(",")]
        else:
            p = line.split()
            
        if len(p) < 4: 
            continue
            
        try: 
            cm = float(p[2]) if p[2] not in ("0", ".", "NA", "") else 0.0
        except ValueError: 
            cm = 0.0
            
        try: 
            bp = int(float(p[3]))
        except (ValueError, TypeError): 
            continue
            
        # 3. Nettoyer le nom du chromosome (enlever le .0 si présent : "1.0" -> "1")
        chr_name = str(p[0]).strip().replace('.0', '')
        
        rows.append({"CHR": chr_name, "SNP": p[1], "CM": cm, "BP": bp})
        
    if not rows: 
        raise ValueError(".map vide ou invalide")
        
    df = pd.DataFrame(rows)
    
    # 4. S'assurer que BP est un entier
    df['BP'] = df['BP'].astype(int)
    
    return df
