/*
 * This script contains the language-specific data used by searchtools.js,
 * namely the list of stopwords, stemmer, scorer and splitter.
 */

var stopwords = ["\u0430", "\u0431\u0435\u0437", "\u0431\u043e\u043b\u0435\u0435", "\u0431\u043e\u043b\u044c\u0448\u0435", "\u0431\u0443\u0434\u0435\u0442", "\u0431\u0443\u0434\u0442\u043e", "\u0431\u044b", "\u0431\u044b\u043b", "\u0431\u044b\u043b\u0430", "\u0431\u044b\u043b\u0438", "\u0431\u044b\u043b\u043e", "\u0431\u044b\u0442\u044c", "\u0432", "\u0432\u0430\u043c", "\u0432\u0430\u0441", "\u0432\u0434\u0440\u0443\u0433", "\u0432\u0435\u0434\u044c", "\u0432\u043e", "\u0432\u043e\u0442", "\u0432\u043f\u0440\u043e\u0447\u0435\u043c", "\u0432\u0441\u0435", "\u0432\u0441\u0435\u0433\u0434\u0430", "\u0432\u0441\u0435\u0433\u043e", "\u0432\u0441\u0435\u0445", "\u0432\u0441\u044e", "\u0432\u044b", "\u0433\u0434\u0435", "\u0433\u043e\u0432\u043e\u0440\u0438\u043b", "\u0434\u0430", "\u0434\u0430\u0436\u0435", "\u0434\u0432\u0430", "\u0434\u043b\u044f", "\u0434\u043e", "\u0434\u0440\u0443\u0433\u043e\u0439", "\u0435\u0433\u043e", "\u0435\u0435", "\u0435\u0439", "\u0435\u043c\u0443", "\u0435\u0441\u043b\u0438", "\u0435\u0441\u0442\u044c", "\u0435\u0449\u0435", "\u0436", "\u0436\u0435", "\u0436\u0438\u0437\u043d\u044c", "\u0437\u0430", "\u0437\u0430\u0447\u0435\u043c", "\u0437\u0434\u0435\u0441\u044c", "\u0438", "\u0438\u0437", "\u0438\u043b\u0438", "\u0438\u043c", "\u0438\u043d\u043e\u0433\u0434\u0430", "\u0438\u0445", "\u043a", "\u043a\u0430\u0436\u0435\u0442\u0441\u044f", "\u043a\u0430\u043a", "\u043a\u0430\u043a\u0430\u044f", "\u043a\u0430\u043a\u043e\u0439", "\u043a\u043e\u0433\u0434\u0430", "\u043a\u043e\u043d\u0435\u0447\u043d\u043e", "\u043a\u0442\u043e", "\u043a\u0443\u0434\u0430", "\u043b\u0438", "\u043b\u0443\u0447\u0448\u0435", "\u043c\u0435\u0436\u0434\u0443", "\u043c\u0435\u043d\u044f", "\u043c\u043d\u0435", "\u043c\u043d\u043e\u0433\u043e", "\u043c\u043e\u0436\u0435\u0442", "\u043c\u043e\u0436\u043d\u043e", "\u043c\u043e\u0439", "\u043c\u043e\u044f", "\u043c\u044b", "\u043d\u0430", "\u043d\u0430\u0434", "\u043d\u0430\u0434\u043e", "\u043d\u0430\u043a\u043e\u043d\u0435\u0446", "\u043d\u0430\u0441", "\u043d\u0435", "\u043d\u0435\u0433\u043e", "\u043d\u0435\u0435", "\u043d\u0435\u0439", "\u043d\u0435\u043b\u044c\u0437\u044f", "\u043d\u0435\u0442", "\u043d\u0438", "\u043d\u0438\u0431\u0443\u0434\u044c", "\u043d\u0438\u043a\u043e\u0433\u0434\u0430", "\u043d\u0438\u043c", "\u043d\u0438\u0445", "\u043d\u0438\u0447\u0435\u0433\u043e", "\u043d\u043e", "\u043d\u0443", "\u043e", "\u043e\u0431", "\u043e\u0434\u0438\u043d", "\u043e\u043d", "\u043e\u043d\u0430", "\u043e\u043d\u0438", "\u043e\u043f\u044f\u0442\u044c", "\u043e\u0442", "\u043f\u0435\u0440\u0435\u0434", "\u043f\u043e", "\u043f\u043e\u0434", "\u043f\u043e\u0441\u043b\u0435", "\u043f\u043e\u0442\u043e\u043c", "\u043f\u043e\u0442\u043e\u043c\u0443", "\u043f\u043e\u0447\u0442\u0438", "\u043f\u0440\u0438", "\u043f\u0440\u043e", "\u0440\u0430\u0437", "\u0440\u0430\u0437\u0432\u0435", "\u0441", "\u0441\u0430\u043c", "\u0441\u0432\u043e\u044e", "\u0441\u0435\u0431\u0435", "\u0441\u0435\u0431\u044f", "\u0441\u0435\u0433\u043e\u0434\u043d\u044f", "\u0441\u0435\u0439\u0447\u0430\u0441", "\u0441\u043a\u0430\u0437\u0430\u043b", "\u0441\u043a\u0430\u0437\u0430\u043b\u0430", "\u0441\u043a\u0430\u0437\u0430\u0442\u044c", "\u0441\u043e", "\u0441\u043e\u0432\u0441\u0435\u043c", "\u0442\u0430\u043a", "\u0442\u0430\u043a\u043e\u0439", "\u0442\u0430\u043c", "\u0442\u0435\u0431\u044f", "\u0442\u0435\u043c", "\u0442\u0435\u043f\u0435\u0440\u044c", "\u0442\u043e", "\u0442\u043e\u0433\u0434\u0430", "\u0442\u043e\u0433\u043e", "\u0442\u043e\u0436\u0435", "\u0442\u043e\u043b\u044c\u043a\u043e", "\u0442\u043e\u043c", "\u0442\u043e\u0442", "\u0442\u0440\u0438", "\u0442\u0443\u0442", "\u0442\u044b", "\u0443", "\u0443\u0436", "\u0443\u0436\u0435", "\u0445\u043e\u0440\u043e\u0448\u043e", "\u0445\u043e\u0442\u044c", "\u0447\u0435\u0433\u043e", "\u0447\u0435\u043b\u043e\u0432\u0435\u043a", "\u0447\u0435\u043c", "\u0447\u0435\u0440\u0435\u0437", "\u0447\u0442\u043e", "\u0447\u0442\u043e\u0431", "\u0447\u0442\u043e\u0431\u044b", "\u0447\u0443\u0442\u044c", "\u044d\u0442\u0438", "\u044d\u0442\u043e\u0433\u043e", "\u044d\u0442\u043e\u0439", "\u044d\u0442\u043e\u043c", "\u044d\u0442\u043e\u0442", "\u044d\u0442\u0443", "\u044f"];


/* Non-minified version is copied as a separate JS file, if available */
/**@constructor*/
BaseStemmer = function() {
    this.setCurrent = function(value) {
        this.current = value;
        this.cursor = 0;
        this.limit = this.current.length;
        this.limit_backward = 0;
        this.bra = this.cursor;
        this.ket = this.limit;
    };

    this.getCurrent = function() {
        return this.current;
    };

    this.copy_from = function(other) {
        this.current          = other.current;
        this.cursor           = other.cursor;
        this.limit            = other.limit;
        this.limit_backward   = other.limit_backward;
        this.bra              = other.bra;
        this.ket              = other.ket;
    };

    this.in_grouping = function(s, min, max) {
        if (this.cursor >= this.limit) return false;
        var ch = this.current.charCodeAt(this.cursor);
        if (ch > max || ch < min) return false;
        ch -= min;
        if ((s[ch >>> 3] & (0x1 << (ch & 0x7))) == 0) return false;
        this.cursor++;
        return true;
    };

    this.in_grouping_b = function(s, min, max) {
        if (this.cursor <= this.limit_backward) return false;
        var ch = this.current.charCodeAt(this.cursor - 1);
        if (ch > max || ch < min) return false;
        ch -= min;
        if ((s[ch >>> 3] & (0x1 << (ch & 0x7))) == 0) return false;
        this.cursor--;
        return true;
    };

    this.out_grouping = function(s, min, max) {
        if (this.cursor >= this.limit) return false;
        var ch = this.current.charCodeAt(this.cursor);
        if (ch > max || ch < min) {
            this.cursor++;
            return true;
        }
        ch -= min;
        if ((s[ch >>> 3] & (0X1 << (ch & 0x7))) == 0) {
            this.cursor++;
            return true;
        }
        return false;
    };

    this.out_grouping_b = function(s, min, max) {
        if (this.cursor <= this.limit_backward) return false;
        var ch = this.current.charCodeAt(this.cursor - 1);
        if (ch > max || ch < min) {
            this.cursor--;
            return true;
        }
        ch -= min;
        if ((s[ch >>> 3] & (0x1 << (ch & 0x7))) == 0) {
            this.cursor--;
            return true;
        }
        return false;
    };

    this.eq_s = function(s)
    {
        if (this.limit - this.cursor < s.length) return false;
        if (this.current.slice(this.cursor, this.cursor + s.length) != s)
        {
            return false;
        }
        this.cursor += s.length;
        return true;
    };

    this.eq_s_b = function(s)
    {
        if (this.cursor - this.limit_backward < s.length) return false;
        if (this.current.slice(this.cursor - s.length, this.cursor) != s)
        {
            return false;
        }
        this.cursor -= s.length;
        return true;
    };

    /** @return {number} */ this.find_among = function(v)
    {
        var i = 0;
        var j = v.length;

        var c = this.cursor;
        var l = this.limit;

        var common_i = 0;
        var common_j = 0;

        var first_key_inspected = false;

        while (true)
        {
            var k = i + ((j - i) >>> 1);
            var diff = 0;
            var common = common_i < common_j ? common_i : common_j; // smaller
            // w[0]: string, w[1]: substring_i, w[2]: result, w[3]: function (optional)
            var w = v[k];
            var i2;
            for (i2 = common; i2 < w[0].length; i2++)
            {
                if (c + common == l)
                {
                    diff = -1;
                    break;
                }
                diff = this.current.charCodeAt(c + common) - w[0].charCodeAt(i2);
                if (diff != 0) break;
                common++;
            }
            if (diff < 0)
            {
                j = k;
                common_j = common;
            }
            else
            {
                i = k;
                common_i = common;
            }
            if (j - i <= 1)
            {
                if (i > 0) break; // v->s has been inspected
                if (j == i) break; // only one item in v

                // - but now we need to go round once more to get
                // v->s inspected. This looks messy, but is actually
                // the optimal approach.

                if (first_key_inspected) break;
                first_key_inspected = true;
            }
        }
        do {
            var w = v[i];
            if (common_i >= w[0].length)
            {
                this.cursor = c + w[0].length;
                if (w.length < 4) return w[2];
                var res = w[3](this);
                this.cursor = c + w[0].length;
                if (res) return w[2];
            }
            i = w[1];
        } while (i >= 0);
        return 0;
    };

    // find_among_b is for backwards processing. Same comments apply
    this.find_among_b = function(v)
    {
        var i = 0;
        var j = v.length

        var c = this.cursor;
        var lb = this.limit_backward;

        var common_i = 0;
        var common_j = 0;

        var first_key_inspected = false;

        while (true)
        {
            var k = i + ((j - i) >> 1);
            var diff = 0;
            var common = common_i < common_j ? common_i : common_j;
            var w = v[k];
            var i2;
            for (i2 = w[0].length - 1 - common; i2 >= 0; i2--)
            {
                if (c - common == lb)
                {
                    diff = -1;
                    break;
                }
                diff = this.current.charCodeAt(c - 1 - common) - w[0].charCodeAt(i2);
                if (diff != 0) break;
                common++;
            }
            if (diff < 0)
            {
                j = k;
                common_j = common;
            }
            else
            {
                i = k;
                common_i = common;
            }
            if (j - i <= 1)
            {
                if (i > 0) break;
                if (j == i) break;
                if (first_key_inspected) break;
                first_key_inspected = true;
            }
        }
        do {
            var w = v[i];
            if (common_i >= w[0].length)
            {
                this.cursor = c - w[0].length;
                if (w.length < 4) return w[2];
                var res = w[3](this);
                this.cursor = c - w[0].length;
                if (res) return w[2];
            }
            i = w[1];
        } while (i >= 0);
        return 0;
    };

    /* to replace chars between c_bra and c_ket in this.current by the
     * chars in s.
     */
    this.replace_s = function(c_bra, c_ket, s)
    {
        var adjustment = s.length - (c_ket - c_bra);
        this.current = this.current.slice(0, c_bra) + s + this.current.slice(c_ket);
        this.limit += adjustment;
        if (this.cursor >= c_ket) this.cursor += adjustment;
        else if (this.cursor > c_bra) this.cursor = c_bra;
        return adjustment;
    };

    this.slice_check = function()
    {
        if (this.bra < 0 ||
            this.bra > this.ket ||
            this.ket > this.limit ||
            this.limit > this.current.length)
        {
            return false;
        }
        return true;
    };

    this.slice_from = function(s)
    {
        var result = false;
        if (this.slice_check())
        {
            this.replace_s(this.bra, this.ket, s);
            result = true;
        }
        return result;
    };

    this.slice_del = function()
    {
        return this.slice_from("");
    };

    this.insert = function(c_bra, c_ket, s)
    {
        var adjustment = this.replace_s(c_bra, c_ket, s);
        if (c_bra <= this.bra) this.bra += adjustment;
        if (c_bra <= this.ket) this.ket += adjustment;
    };

    this.slice_to = function()
    {
        var result = '';
        if (this.slice_check())
        {
            result = this.current.slice(this.bra, this.ket);
        }
        return result;
    };

    this.assign_to = function()
    {
        return this.current.slice(0, this.limit);
    };
};

// Generated by Snowball 2.1.0 - https://snowballstem.org/

/**@constructor*/
RussianStemmer = function() {
    var base = new BaseStemmer();
    /** @const */ var a_0 = [
        ["\u0432", -1, 1],
        ["\u0438\u0432", 0, 2],
        ["\u044B\u0432", 0, 2],
        ["\u0432\u0448\u0438", -1, 1],
        ["\u0438\u0432\u0448\u0438", 3, 2],
        ["\u044B\u0432\u0448\u0438", 3, 2],
        ["\u0432\u0448\u0438\u0441\u044C", -1, 1],
        ["\u0438\u0432\u0448\u0438\u0441\u044C", 6, 2],
        ["\u044B\u0432\u0448\u0438\u0441\u044C", 6, 2]
    ];

    /** @const */ var a_1 = [
        ["\u0435\u0435", -1, 1],
        ["\u0438\u0435", -1, 1],
        ["\u043E\u0435", -1, 1],
        ["\u044B\u0435", -1, 1],
        ["\u0438\u043C\u0438", -1, 1],
        ["\u044B\u043C\u0438", -1, 1],
        ["\u0435\u0439", -1, 1],
        ["\u0438\u0439", -1, 1],
        ["\u043E\u0439", -1, 1],
        ["\u044B\u0439", -1, 1],
        ["\u0435\u043C", -1, 1],
        ["\u0438\u043C", -1, 1],
        ["\u043E\u043C", -1, 1],
        ["\u044B\u043C", -1, 1],
        ["\u0435\u0433\u043E", -1, 1],
        ["\u043E\u0433\u043E", -1, 1],
        ["\u0435\u043C\u0443", -1, 1],
        ["\u043E\u043C\u0443", -1, 1],
        ["\u0438\u0445", -1, 1],
        ["\u044B\u0445", -1, 1],
        ["\u0435\u044E", -1, 1],
        ["\u043E\u044E", -1, 1],
        ["\u0443\u044E", -1, 1],
        ["\u044E\u044E", -1, 1],
        ["\u0430\u044F", -1, 1],
        ["\u044F\u044F", -1, 1]
    ];

    /** @const */ var a_2 = [
        ["\u0435\u043C", -1, 1],
        ["\u043D\u043D", -1, 1],
        ["\u0432\u0448", -1, 1],
        ["\u0438\u0432\u0448", 2, 2],
        ["\u044B\u0432\u0448", 2, 2],
        ["\u0449", -1, 1],
        ["\u044E\u0449", 5, 1],
        ["\u0443\u044E\u0449", 6, 2]
    ];

    /** @const */ var a_3 = [
        ["\u0441\u044C", -1, 1],
        ["\u0441\u044F", -1, 1]
    ];

    /** @const */ var a_4 = [
        ["\u043B\u0430", -1, 1],
        ["\u0438\u043B\u0430", 0, 2],
        ["\u044B\u043B\u0430", 0, 2],
        ["\u043D\u0430", -1, 1],
        ["\u0435\u043D\u0430", 3, 2],
        ["\u0435\u0442\u0435", -1, 1],
        ["\u0438\u0442\u0435", -1, 2],
        ["\u0439\u0442\u0435", -1, 1],
        ["\u0435\u0439\u0442\u0435", 7, 2],
        ["\u0443\u0439\u0442\u0435", 7, 2],
        ["\u043B\u0438", -1, 1],
        ["\u0438\u043B\u0438", 10, 2],
        ["\u044B\u043B\u0438", 10, 2],
        ["\u0439", -1, 1],
        ["\u0435\u0439", 13, 2],
        ["\u0443\u0439", 13, 2],
        ["\u043B", -1, 1],
        ["\u0438\u043B", 16, 2],
        ["\u044B\u043B", 16, 2],
        ["\u0435\u043C", -1, 1],
        ["\u0438\u043C", -1, 2],
        ["\u044B\u043C", -1, 2],
        ["\u043D", -1, 1],
        ["\u0435\u043D", 22, 2],
        ["\u043B\u043E", -1, 1],
        ["\u0438\u043B\u043E", 24, 2],
        ["\u044B\u043B\u043E", 24, 2],
        ["\u043D\u043E", -1, 1],
        ["\u0435\u043D\u043E", 27, 2],
        ["\u043D\u043D\u043E", 27, 1],
        ["\u0435\u0442", -1, 1],
        ["\u0443\u0435\u0442", 30, 2],
        ["\u0438\u0442", -1, 2],
        ["\u044B\u0442", -1, 2],
        ["\u044E\u0442", -1, 1],
        ["\u0443\u044E\u0442", 34, 2],
        ["\u044F\u0442", -1, 2],
        ["\u043D\u044B", -1, 1],
        ["\u0435\u043D\u044B", 37, 2],
        ["\u0442\u044C", -1, 1],
        ["\u0438\u0442\u044C", 39, 2],
        ["\u044B\u0442\u044C", 39, 2],
        ["\u0435\u0448\u044C", -1, 1],
        ["\u0438\u0448\u044C", -1, 2],
        ["\u044E", -1, 2],
        ["\u0443\u044E", 44, 2]
    ];

    /** @const */ var a_5 = [
        ["\u0430", -1, 1],
        ["\u0435\u0432", -1, 1],
        ["\u043E\u0432", -1, 1],
        ["\u0435", -1, 1],
        ["\u0438\u0435", 3, 1],
        ["\u044C\u0435", 3, 1],
        ["\u0438", -1, 1],
        ["\u0435\u0438", 6, 1],
        ["\u0438\u0438", 6, 1],
        ["\u0430\u043C\u0438", 6, 1],
        ["\u044F\u043C\u0438", 6, 1],
        ["\u0438\u044F\u043C\u0438", 10, 1],
        ["\u0439", -1, 1],
        ["\u0435\u0439", 12, 1],
        ["\u0438\u0435\u0439", 13, 1],
        ["\u0438\u0439", 12, 1],
        ["\u043E\u0439", 12, 1],
        ["\u0430\u043C", -1, 1],
        ["\u0435\u043C", -1, 1],
        ["\u0438\u0435\u043C", 18, 1],
        ["\u043E\u043C", -1, 1],
        ["\u044F\u043C", -1, 1],
        ["\u0438\u044F\u043C", 21, 1],
        ["\u043E", -1, 1],
        ["\u0443", -1, 1],
        ["\u0430\u0445", -1, 1],
        ["\u044F\u0445", -1, 1],
        ["\u0438\u044F\u0445", 26, 1],
        ["\u044B", -1, 1],
        ["\u044C", -1, 1],
        ["\u044E", -1, 1],
        ["\u0438\u044E", 30, 1],
        ["\u044C\u044E", 30, 1],
        ["\u044F", -1, 1],
        ["\u0438\u044F", 33, 1],
        ["\u044C\u044F", 33, 1]
    ];

    /** @const */ var a_6 = [
        ["\u043E\u0441\u0442", -1, 1],
        ["\u043E\u0441\u0442\u044C", -1, 1]
    ];

    /** @const */ var a_7 = [
        ["\u0435\u0439\u0448\u0435", -1, 1],
        ["\u043D", -1, 2],
        ["\u0435\u0439\u0448", -1, 1],
        ["\u044C", -1, 3]
    ];

    /** @const */ var /** Array<int> */ g_v = [33, 65, 8, 232];

    var /** number */ I_p2 = 0;
    var /** number */ I_pV = 0;


    /** @return {boolean} */
    function r_mark_regions() {
        I_pV = base.limit;
        I_p2 = base.limit;
        var /** number */ v_1 = base.cursor;
        lab0: {
            golab1: while(true)
            {
                lab2: {
                    if (!(base.in_grouping(g_v, 1072, 1103)))
                    {
                        break lab2;
                    }
                    break golab1;
                }
                if (base.cursor >= base.limit)
                {
                    break lab0;
                }
                base.cursor++;
            }
            I_pV = base.cursor;
            golab3: while(true)
            {
                lab4: {
                    if (!(base.out_grouping(g_v, 1072, 1103)))
                    {
                        break lab4;
                    }
                    break golab3;
                }
                if (base.cursor >= base.limit)
                {
                    break lab0;
                }
                base.cursor++;
            }
            golab5: while(true)
            {
                lab6: {
                    if (!(base.in_grouping(g_v, 1072, 1103)))
                    {
                        break lab6;
                    }
                    break golab5;
                }
                if (base.cursor >= base.limit)
                {
                    break lab0;
                }
                base.cursor++;
            }
            golab7: while(true)
            {
                lab8: {
                    if (!(base.out_grouping(g_v, 1072, 1103)))
                    {
                        break lab8;
                    }
                    break golab7;
                }
                if (base.cursor >= base.limit)
                {
                    break lab0;
                }
                base.cursor++;
            }
            I_p2 = base.cursor;
        }
        base.cursor = v_1;
        return true;
    };

    /** @return {boolean} */
    function r_R2() {
        if (!(I_p2 <= base.cursor))
        {
            return false;
        }
        return true;
    };

    /** @return {boolean} */
    function r_perfective_gerund() {
        var /** number */ among_var;
        base.ket = base.cursor;
        among_var = base.find_among_b(a_0);
        if (among_var == 0)
        {
            return false;
        }
        base.bra = base.cursor;
        switch (among_var) {
            case 1:
                lab0: {
                    var /** number */ v_1 = base.limit - base.cursor;
                    lab1: {
                        if (!(base.eq_s_b("\u0430")))
                        {
                            break lab1;
                        }
                        break lab0;
                    }
                    base.cursor = base.limit - v_1;
                    if (!(base.eq_s_b("\u044F")))
                    {
                        return false;
                    }
                }
                if (!base.slice_del())
                {
                    return false;
                }
                break;
            case 2:
                if (!base.slice_del())
                {
                    return false;
                }
                break;
        }
        return true;
    };

    /** @return {boolean} */
    function r_adjective() {
        base.ket = base.cursor;
        if (base.find_among_b(a_1) == 0)
        {
            return false;
        }
        base.bra = base.cursor;
        if (!base.slice_del())
        {
            return false;
        }
        return true;
    };

    /** @return {boolean} */
    function r_adjectival() {
        var /** number */ among_var;
        if (!r_adjective())
        {
            return false;
        }
        var /** number */ v_1 = base.limit - base.cursor;
        lab0: {
            base.ket = base.cursor;
            among_var = base.find_among_b(a_2);
            if (among_var == 0)
            {
                base.cursor = base.limit - v_1;
                break lab0;
            }
            base.bra = base.cursor;
            switch (among_var) {
                case 1:
                    lab1: {
                        var /** number */ v_2 = base.limit - base.cursor;
                        lab2: {
                            if (!(base.eq_s_b("\u0430")))
                            {
                                break lab2;
                            }
                            break lab1;
                        }
                        base.cursor = base.limit - v_2;
                        if (!(base.eq_s_b("\u044F")))
                        {
                            base.cursor = base.limit - v_1;
                            break lab0;
                        }
                    }
                    if (!base.slice_del())
                    {
                        return false;
                    }
                    break;
                case 2:
                    if (!base.slice_del())
                    {
                        return false;
                    }
                    break;
            }
        }
        return true;
    };

    /** @return {boolean} */
    function r_reflexive() {
        base.ket = base.cursor;
        if (base.find_among_b(a_3) == 0)
        {
            return false;
        }
        base.bra = base.cursor;
        if (!base.slice_del())
        {
            return false;
        }
        return true;
    };

    /** @return {boolean} */
    function r_verb() {
        var /** number */ among_var;
        base.ket = base.cursor;
        among_var = base.find_among_b(a_4);
        if (among_var == 0)
        {
            return false;
        }
        base.bra = base.cursor;
        switch (among_var) {
            case 1:
                lab0: {
                    var /** number */ v_1 = base.limit - base.cursor;
                    lab1: {
                        if (!(base.eq_s_b("\u0430")))
                        {
                            break lab1;
                        }
                        break lab0;
                    }
                    base.cursor = base.limit - v_1;
                    if (!(base.eq_s_b("\u044F")))
                    {
                        return false;
                    }
                }
                if (!base.slice_del())
                {
                    return false;
                }
                break;
            case 2:
                if (!base.slice_del())
                {
                    return false;
                }
                break;
        }
        return true;
    };

    /** @return {boolean} */
    function r_noun() {
        base.ket = base.cursor;
        if (base.find_among_b(a_5) == 0)
        {
            return false;
        }
        base.bra = base.cursor;
        if (!base.slice_del())
        {
            return false;
        }
        return true;
    };

    /** @return {boolean} */
    function r_derivational() {
        base.ket = base.cursor;
        if (base.find_among_b(a_6) == 0)
        {
            return false;
        }
        base.bra = base.cursor;
        if (!r_R2())
        {
            return false;
        }
        if (!base.slice_del())
        {
            return false;
        }
        return true;
    };

    /** @return {boolean} */
    function r_tidy_up() {
        var /** number */ among_var;
        base.ket = base.cursor;
        among_var = base.find_among_b(a_7);
        if (among_var == 0)
        {
            return false;
        }
        base.bra = base.cursor;
        switch (among_var) {
            case 1:
                if (!base.slice_del())
                {
                    return false;
                }
                base.ket = base.cursor;
                if (!(base.eq_s_b("\u043D")))
                {
                    return false;
                }
                base.bra = base.cursor;
                if (!(base.eq_s_b("\u043D")))
                {
                    return false;
                }
                if (!base.slice_del())
                {
                    return false;
                }
                break;
            case 2:
                if (!(base.eq_s_b("\u043D")))
                {
                    return false;
                }
                if (!base.slice_del())
                {
                    return false;
                }
                break;
            case 3:
                if (!base.slice_del())
                {
                    return false;
                }
                break;
        }
        return true;
    };

    this.stem = /** @return {boolean} */ function() {
        var /** number */ v_1 = base.cursor;
        lab0: {
            while(true)
            {
                var /** number */ v_2 = base.cursor;
                lab1: {
                    golab2: while(true)
                    {
                        var /** number */ v_3 = base.cursor;
                        lab3: {
                            base.bra = base.cursor;
                            if (!(base.eq_s("\u0451")))
                            {
                                break lab3;
                            }
                            base.ket = base.cursor;
                            base.cursor = v_3;
                            break golab2;
                        }
                        base.cursor = v_3;
                        if (base.cursor >= base.limit)
                        {
                            break lab1;
                        }
                        base.cursor++;
                    }
                    if (!base.slice_from("\u0435"))
                    {
                        return false;
                    }
                    continue;
                }
                base.cursor = v_2;
                break;
            }
        }
        base.cursor = v_1;
        r_mark_regions();
        base.limit_backward = base.cursor; base.cursor = base.limit;
        if (base.cursor < I_pV)
        {
            return false;
        }
        var /** number */ v_6 = base.limit_backward;
        base.limit_backward = I_pV;
        var /** number */ v_7 = base.limit - base.cursor;
        lab4: {
            lab5: {
                var /** number */ v_8 = base.limit - base.cursor;
                lab6: {
                    if (!r_perfective_gerund())
                    {
                        break lab6;
                    }
                    break lab5;
                }
                base.cursor = base.limit - v_8;
                var /** number */ v_9 = base.limit - base.cursor;
                lab7: {
                    if (!r_reflexive())
                    {
                        base.cursor = base.limit - v_9;
                        break lab7;
                    }
                }
                lab8: {
                    var /** number */ v_10 = base.limit - base.cursor;
                    lab9: {
                        if (!r_adjectival())
                        {
                            break lab9;
                        }
                        break lab8;
                    }
                    base.cursor = base.limit - v_10;
                    lab10: {
                        if (!r_verb())
                        {
                            break lab10;
                        }
                        break lab8;
                    }
                    base.cursor = base.limit - v_10;
                    if (!r_noun())
                    {
                        break lab4;
                    }
                }
            }
        }
        base.cursor = base.limit - v_7;
        var /** number */ v_11 = base.limit - base.cursor;
        lab11: {
            base.ket = base.cursor;
            if (!(base.eq_s_b("\u0438")))
            {
                base.cursor = base.limit - v_11;
                break lab11;
            }
            base.bra = base.cursor;
            if (!base.slice_del())
            {
                return false;
            }
        }
        var /** number */ v_12 = base.limit - base.cursor;
        r_derivational();
        base.cursor = base.limit - v_12;
        var /** number */ v_13 = base.limit - base.cursor;
        r_tidy_up();
        base.cursor = base.limit - v_13;
        base.limit_backward = v_6;
        base.cursor = base.limit_backward;
        return true;
    };

    /**@return{string}*/
    this['stemWord'] = function(/**string*/word) {
        base.setCurrent(word);
        this.stem();
        return base.getCurrent();
    };
};

Stemmer = RussianStemmer;
