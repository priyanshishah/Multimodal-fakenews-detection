function l = split(d,s)
l = {};
while (length(s) > 0)
    [t,s] = strtok(s,d);
    l = {l{:}, t};
end
