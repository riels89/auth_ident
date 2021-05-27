#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>		/* getopt() */

static enum language {C, JAVA, PYTHON};

// Order of mapping indexes: buffer, alphabet, operators, operators_2, operators_3, keywords, top_n_idenfifiers
/* buffer = [
*    0,  # unknown
*   1,  # start
*   2,  # end
*   3,  # space
*   4,  # tab
*   5,  # newline
*   6,  # integer
*   7,  # floating
*   8,  # char
*   9.  # string
*   10   # preprocessor
] */
static char alphabet[] = "_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
static const char * ops[] = {
//ops1
    "{", "}", "[", "]", "(", ")",
    ";", "?", "~", ",", "@", "<",
    ":", ".", "-", "+", "*", "/",
    "%", "^", "&", "|", "=", "!",
    ">",
//ops2
    "<:", "<%", "<=", "<<", ":>",
    "::", ".*", "->", "-=", "--",
    "+=", "++", "*=", "/=", "%>",
    "%=", "^=", "&=", "&&", "|=",
    "||", "==", "!=", ">=", ">>",
//ops3
    "...", "<=>", "->*", "<<="
};

#define MAXCHARS 20
#define MAXKEYWORDS 100
static char keywords[MAXKEYWORDS][MAXCHARS];
static size_t num_keywords;

static char (*reserved_identifiers)[MAXCHARS];
static int num_reserved_identifiers;

char* get_username(char *filename) {
    if(!filename) {
        printf("get_username received a null filename");
        exit(1);
    }
    char *username;
    char *filename_copy = malloc(strlen(filename) + 1) ;
    strcpy(filename_copy, filename);
    if(!filename_copy) {
        printf("Filename copy was not created correctly");
        exit(1);
    }
    strtok(filename_copy, "/");
    for(int i =0; i<3; i++) {
        strtok(NULL, "/");
    }
    username = strtok(NULL, "/");
    free(filename_copy);
    return username;
}

#define BUFFER_SIZE 300
void process(FILE *input_file, FILE *output_file) {
    int alphabet_index_buffer = 13;
    int op_index_buffer = alphabet_index_buffer + 26 * 2 + 10 + 1;
    int num_opereators = (sizeof(ops) / sizeof(ops[0]));
    int keyword_index_buffer = alphabet_index_buffer + num_opereators;
    int reserved_identifiers_buffer = keyword_index_buffer + num_keywords;

    printf("size of ops: %ld\n", sizeof(ops) / sizeof(ops[0]));
    printf("op_index_buffer: %d\n", op_index_buffer);
    printf("keyword_index_buffer: %d\n", keyword_index_buffer);
    printf("reserved_identifiers_buffer: %d\n", reserved_identifiers_buffer);

    long progress = 0;
    static char buffer[BUFFER_SIZE];    // use buffer as multi-char lookahead.
    char *curr_filename;
    char *curr_username;
    char *curr_token;
    fprintf(output_file, "username|filename|file_content\n");
    while(fgets(buffer, BUFFER_SIZE, input_file)) {

        char *class = strtok(buffer, ",");
        if(strcmp(class, "filename") == 0) {
            progress++;
            printf("\rWorking on file number: %ld", progress);
            fflush(stdout);
            //printf("class: %s\n", class);
            if (progress != 1)
                fprintf(output_file, "]\"\n");
            curr_filename = strtok(NULL, ",");
            curr_filename[strlen(curr_filename) - 1] = 0;
            //printf("Filename: %s\n", curr_filename);
            curr_username = get_username(curr_filename);
            //printf("usernme: %s\n", curr_username);
            fprintf(output_file,"%s|%s|\"[", curr_username, curr_filename);
        }
        else if (strcmp(class, "s") == 0) {
            fprintf(output_file, "3, ");
        }
        else if (strcmp(class, "t") == 0) {
            fprintf(output_file, "4, ");
        }
        else if (strcmp(class, "c") == 0) {
            fprintf(output_file, "5, ");
        }
        else if (strcmp(class, "lc") == 0) {
            fprintf(output_file, "6, ");
        }
        else if (strcmp(class, "newline") == 0) {
            fprintf(output_file, "7, ");
        }
        else if (strcmp(class, "integer") == 0) {
            fprintf(output_file, "8, ");
        }
        else if (strcmp(class, "floating") == 0) {
            fprintf(output_file, "9, ");
        }
        else if (strcmp(class, "char") == 0) {
            fprintf(output_file, "10, ");
        }
        else if (strcmp(class, "string") == 0) {
            fprintf(output_file, "11, ");
        }
        else if (strcmp(class, "preprocessor") == 0) {
            fprintf(output_file, "12, ");
        }
        else if (strcmp(class, "operator") == 0) {
            curr_token = strtok(NULL, ",");
            for(int i = 0; i < num_opereators; i++) {
                if(strcmp(curr_token, ops[i]) == 0) {
                    fprintf(output_file, "%d, ", i + op_index_buffer);
                }
            }
        }
        else if (strcmp(class, "keyword") == 0) {
            curr_token = strtok(NULL, ",");
            for(int i = 0; i < num_keywords; i++) {
                if(strcmp(curr_token, keywords[i]) == 0) {
                    fprintf(output_file, "%d, ", i + keyword_index_buffer);
                }
            }
        }
        else if (strcmp(class, "identifier") == 0) {
            curr_token = strtok(NULL, ",");
            int i = 0;
            for(; i < num_reserved_identifiers; i++) {
                if(strcmp(curr_token, reserved_identifiers[i]) == 0) {
                    fprintf(output_file, "%d, ", i + reserved_identifiers_buffer);
                }
            }
            if (i == num_reserved_identifiers) {
                int index;
                for(int cc = 0; cc < strlen(curr_token); cc++) {
                    const char *ptr = strchr(alphabet, curr_token[cc]);
                    if(ptr) {
                         index = ptr - alphabet + alphabet_index_buffer;
                    }
                    fprintf(output_file, "%d, ", index);
                }
            }
        }
    }
    fprintf(output_file, "]\"\n");
}

void load_words(FILE *f, size_t rows, size_t cols, char buffer[rows][cols], size_t* num_lines) {
    int i = 0;
    int j = 0;
    *num_lines = 0;
    while(i < rows && fgets(buffer[i], cols, f)) {
        char *p = buffer[i];
        while(*p != '\n' && j < cols) {
            p++;
            j++;
        }
        *p = 0; // Set to null chart
        i++;
        j = 0;
        *num_lines += 1;
        printf("Line %d: %s\n", i, buffer[i - 1]);
    }
}

void load_identifiers(FILE* f) {
    reserved_identifiers = malloc(sizeof(*reserved_identifiers) * num_reserved_identifiers);
    if(!reserved_identifiers) {
        printf("Error allocating reserved_identifiers");
        exit(1);
    }

    size_t placeholder = 0;
    printf("num_reserved_identifiers: %d, cols: %d", num_reserved_identifiers, MAXCHARS);
    load_words(f, num_reserved_identifiers, MAXCHARS, reserved_identifiers, &placeholder);
}

void load_keywords(enum language lang) {
    FILE *keyword_file;
    switch (lang) {
        case (PYTHON):
            printf("Python not yet implemented");
            exit(1);
            break;
        case (JAVA):
            keyword_file = fopen("auth_ident/preprocessing/java.kw", "r");
            break;
        case (C):
            keyword_file = fopen("auth_ident/preprocessing/c++20.kw", "r");
            break;
        default:
            keyword_file = fopen("auth_ident/preprocessing/c++20.kw", "r");
            break;
    }
    if(!keyword_file) {
        printf("Failed to load keyword file\n");
        exit(1);
    }
   load_words(keyword_file, MAXKEYWORDS, MAXCHARS, keywords, &num_keywords); 
   fclose(keyword_file);
}


int main(int argc, char *argv[])
{
    extern char *optarg;
    extern int opterr;
    extern int optind;
    int option;
    char const *opt_str = "n:t:";

    while ((option = getopt(argc, argv, opt_str)) != EOF) {
        switch (option) {
            case('n'):
                num_reserved_identifiers = atoi(optarg);
                break;
        }
    }
    printf("Getting %d reserved identifiers\n", num_reserved_identifiers);

	char input_postfix[] = "_tokenized_csv.csv";
	char output_postfix[] = "_encoded.csv";
    char reserved_identifiers_postfix[] = "_top_identifiers.txt";

    char *base = argv[optind];
   	char* input_filename = (char*) malloc(strlen(base) + strlen(input_postfix) + 1);
	char* output_filename = (char*) malloc(strlen(base) + strlen(output_postfix) + 1);
	char* reserved_identifiers_filename = (char*) malloc(strlen(base) + strlen(reserved_identifiers_postfix) + 1);
	if (!input_filename || !output_filename || !reserved_identifiers_filename) {
        free(input_filename);
        free(output_filename);
        free(reserved_identifiers_filename);
		exit(1);
	}
	strcpy(input_filename, base);
	strcat(input_filename, input_postfix);
	printf("READING FILE: %s\n", input_filename);

	strcpy(output_filename, base);
	strcat(output_filename, output_postfix);
	printf("OUTPUT FILE: %s\n", output_filename);

    int found_underscore = 0;
    for(int i=strlen(base) - 1; i >= 0; i--) {
        if (found_underscore) {
            reserved_identifiers_filename[i] = base[i];
        }
        else {
            if (base[i] == '_') {
                found_underscore = 1;
            }
        }
    }
	strcat(reserved_identifiers_filename, reserved_identifiers_postfix);
	printf("IDENTIFIERS FILE: %s\n", reserved_identifiers_filename);

	FILE *input_file = fopen(input_filename, "r");
	FILE *output_file = fopen(output_filename, "w");
	FILE *reserved_identifiers_file = fopen(reserved_identifiers_filename, "r");
	if (!input_file || !output_file || !reserved_identifiers_filename) {
        free(input_filename);
        free(output_filename);
        free(reserved_identifiers_filename);
        fclose(input_file);
        fclose(output_file);
        fclose(reserved_identifiers_file);
        printf("Failed to open file");
		exit(1);
	}
    //output_file = stdout;

    load_keywords(C);
    load_identifiers(reserved_identifiers_file);
    process(input_file, output_file);

    free(reserved_identifiers);

    fclose(input_file);
    fclose(output_file);
    fclose(reserved_identifiers_file);
    free(input_filename);
    free(output_filename);
    free(reserved_identifiers_filename);
    
    return 0;
}
