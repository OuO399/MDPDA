
lst_dimension=[50,40]
projects = ["ambari", "ant", "aptoide", "camel", "cassandra", "egeria", "felix", "jackrabbit", "jenkins",
                "lucene-solr"]
for dimension in lst_dimension:
    for project in projects:
        read_file=open("./{}/{}.txt".format(dimension,project),"r+").readlines()
        for i in range(0,len(read_file)):
            if len(read_file[i].split(" ")) != 2:
                read_file=read_file[i:]
                break
        write_file=open("./{}/{}.txt".format(dimension,project),"w+")
        write_file.write("{} {}\n".format(len(read_file),dimension))
        for line in read_file:
            write_file.write(line)