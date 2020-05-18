
%

function Culture=AdjustCulture(Culture,spop)

    n=numel(spop);
    nVar=numel(spop(1).Position);
    
    for i=1:n
        if spop(i).Cost<Culture.Situational.Cost
            Culture.Situational=spop(i);
        end
        
        for j=1:nVar
            if spop(i).Position(j)<Culture.Normative.Min(j) ...
                    || spop(i).Cost<Culture.Normative.L(j)
                Culture.Normative.Min(j)=spop(i).Position(j);
                Culture.Normative.L(j)=spop(i).Cost;
            end
            if spop(i).Position(j)>Culture.Normative.Max(j) ...
                    || spop(i).Cost<Culture.Normative.U(j)
                Culture.Normative.Max(j)=spop(i).Position(j);
                Culture.Normative.U(j)=spop(i).Cost;
            end
        end
    end

    Culture.Normative.Size=Culture.Normative.Max-Culture.Normative.Min;
    
end